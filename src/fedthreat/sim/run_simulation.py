"""Federated simulation coordinator and experiment engine."""

from typing import Dict, List, Tuple, Any
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from fedthreat.features.spec import FeatureSpec, CanonicalEvent
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.models.mlp import ThreatMLP
from fedthreat.client.trainer import ClientTrainer
from fedthreat.aggregation.fedavg import FedAvgAggregator
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset, partition_non_iid_dirichlet
from fedthreat.metrics import compute_threat_metrics


class FederatedSimulation:
    """Orchestrates non-IID federated training across virtual organizations."""

    def __init__(
        self,
        num_orgs: int = 5,
        alpha: float = 0.5,
        held_out_attack: str = "infiltration",
        held_out_org: int = 0,
        spec_path: str = "configs/feature_spec_v1.yaml",
        seed: int = 42,
    ):
        self.num_orgs = num_orgs
        self.alpha = alpha
        self.held_out_attack = held_out_attack
        self.held_out_org = held_out_org
        self.seed = seed

        # Load spec and extractor
        with open(spec_path, "r") as f:
            spec_dict = yaml.safe_load(f)
        self.spec = FeatureSpec(**spec_dict)
        self.extractor = FeatureExtractor(self.spec)

        self.aggregator = FedAvgAggregator()
        self.clients: Dict[int, ClientTrainer] = {}
        self.test_sets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.held_out_test_set: Tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
        self.global_model = ThreatMLP(input_dim=self.extractor.output_dim, hidden_dims=[128, 64, 32], num_classes=2)

    def prepare_data(self, num_samples: int = 15000) -> None:
        """Generates synthetic data, creates non-IID partitions, and initializes client loaders."""
        events = generate_synthetic_flow_dataset(num_samples=num_samples, seed=self.seed)
        partitions = partition_non_iid_dirichlet(
            events,
            num_orgs=self.num_orgs,
            alpha=self.alpha,
            seed=self.seed,
            held_out_attack=self.held_out_attack,
            held_out_org=self.held_out_org,
        )

        # Prepare a separate held-out attack test set to test transferability
        held_out_events = [e for e in events if e.label == self.held_out_attack]
        X_held = self.extractor.extract_batch(held_out_events)
        y_held = np.ones(len(held_out_events), dtype=np.int64)

        # Compute global normalization statistics from pooled train sets
        pooled_X_list = []
        for org_id, org_events in partitions.items():
            if org_events:
                X_org = self.extractor.extract_batch(org_events)
                pooled_X_list.append(X_org)

        all_X = np.vstack(pooled_X_list)
        norm_mean = np.mean(all_X, axis=0, keepdims=True)
        norm_std = np.std(all_X, axis=0, keepdims=True) + 1e-7

        # Normalize held-out test set
        self.held_out_test_set = ((X_held - norm_mean) / norm_std, y_held)

        for org_id, org_events in partitions.items():
            X_org = self.extractor.extract_batch(org_events)
            y_org = np.array([0 if e.label == "benign" else 1 for e in org_events], dtype=np.int64)

            # Normalize
            X_org = (X_org - norm_mean) / norm_std

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_org, y_org, test_size=0.25, random_state=self.seed, stratify=y_org
            )

            tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr))
            val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))

            tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

            client_model = ThreatMLP(
                input_dim=self.extractor.output_dim, hidden_dims=[128, 64, 32], num_classes=2
            )
            self.clients[org_id] = ClientTrainer(
                client_id=org_id,
                model=client_model,
                train_loader=tr_loader,
                val_loader=val_loader,
                lr=1e-3,
            )
            self.test_sets[org_id] = (X_val, y_val)

    def evaluate_model_on_data(
        self, model: ThreatMLP, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Runs inference on a given dataset and returns security metrics."""
        model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        return compute_threat_metrics(y, probs, preds, target_fpr=0.001)

    def run_local_only_baseline(self, epochs_per_org: int = 10) -> Dict[int, Dict[str, float]]:
        """Trains each org purely on its own local data (no federation) to establish baseline."""
        results = {}
        for org_id, client in self.clients.items():
            init_params = {k: v.detach().cpu().numpy().copy() for k, v in self.global_model.state_dict().items()}
            client.train_round(init_params, epochs=epochs_per_org, mu_fedprox=0.0)
            
            # Evaluate on held-out attack family
            X_held, y_held = self.held_out_test_set
            held_metrics = self.evaluate_model_on_data(client.model, X_held, y_held)
            results[org_id] = held_metrics
        return results

    def run_federated_rounds(
        self,
        num_rounds: int = 10,
        local_epochs: int = 2,
        mu_fedprox: float = 0.0,
        clip_norm: float | None = None,
        noise_multiplier: float = 0.0,
        target_delta: float = 1e-5,
        aggregator: Any | None = None,
        malicious_clients: Dict[int, Any] | None = None,
        max_epsilon: float = 20.0,
    ) -> Dict[str, Any]:
        """Executes federated rounds across all organizations with DP and Byzantine attacks."""
        from fedthreat.client.dp import PrivacyAccountant

        active_aggregator = aggregator if aggregator is not None else self.aggregator
        accountant = PrivacyAccountant(target_delta=target_delta, max_epsilon=max_epsilon)
        global_params = {
            k: v.detach().cpu().numpy().copy() for k, v in self.global_model.state_dict().items()
        }

        history = []

        for r in range(1, num_rounds + 1):
            if noise_multiplier > 0.0 and not accountant.can_afford(noise_multiplier, 1.0):
                print(f"[!] Next round would exceed epsilon={max_epsilon}; halting at round {r}.")
                break

            client_deltas = []
            client_weights = []

            for org_id, client in self.clients.items():
                delta, n_samples, loss = client.train_round(
                    global_params,
                    epochs=local_epochs,
                    mu_fedprox=mu_fedprox,
                    clip_norm=clip_norm,
                    noise_multiplier=noise_multiplier,
                )

                # Check if this client is malicious
                if malicious_clients and org_id in malicious_clients:
                    attacker = malicious_clients[org_id]
                    delta = attacker.poison_update(
                        clean_update=delta,
                        round_num=r,
                        context={"num_clients": len(self.clients), "client_id": org_id},
                    )

                client_deltas.append(delta)
                client_weights.append(float(n_samples))

            # Record round privacy spend
            if noise_multiplier > 0.0:
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=1.0)

            # Aggregate on server using chosen strategy
            global_params = active_aggregator.aggregate(client_deltas, client_weights, global_params)

            # Update global model
            state_dict = {k: torch.tensor(v) for k, v in global_params.items()}
            self.global_model.load_state_dict(state_dict)

            # Evaluate global model on held-out attack
            X_held, y_held = self.held_out_test_set
            held_metrics = self.evaluate_model_on_data(self.global_model, X_held, y_held)
            curr_eps = accountant.get_epsilon()
            history.append({
                "round": r,
                "epsilon": curr_eps,
                "held_out_metrics": held_metrics,
            })

        return {"global_params": global_params, "history": history, "final_epsilon": accountant.get_epsilon()}
