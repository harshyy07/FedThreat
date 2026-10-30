"""Federated client local trainer with FedAvg and FedProx support."""

from typing import Dict, Tuple, Optional
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedthreat.models.mlp import ThreatMLP
from fedthreat.client.dp import add_gaussian_noise, clip_update, per_client_noise_std
from fedthreat.metrics import compute_threat_metrics


class ClientTrainer:
    """Manages local training for an organization node."""

    def __init__(
        self,
        client_id: int,
        model: ThreatMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Returns model parameters as a dictionary of numpy arrays."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Loads numpy parameter dictionary into model."""
        state_dict = {k: torch.tensor(v, device=self.device) for k, v in params.items()}
        self.model.load_state_dict(state_dict)

    def train_round(
        self,
        global_params: Dict[str, np.ndarray],
        epochs: int = 2,
        mu_fedprox: float = 0.0,
        clip_norm: float | None = None,
        noise_multiplier: float = 0.0,
        honest_min: int = 1,
    ) -> Tuple[Dict[str, np.ndarray], int, float]:
        """Runs local epochs starting from global parameters.
        
        Args:
            global_params: Weights distributed by the server for this round.
            epochs: Number of local training epochs.
            mu_fedprox: FedProx proximal term coefficient (0.0 = standard FedAvg).
            clip_norm: Maximum L2 norm of the parameter update delta (bounds sensitivity).
            noise_multiplier: sigma. The SUM of >= honest_min honest clients carries N(0,(sigma*C)^2);
                each client adds N(0,(sigma*C)^2/honest_min). Use honest_min=1 for local-only noise.
            honest_min: number of honest clients assumed in every secure aggregate.
            
        Returns:
            (delta_w, num_samples, avg_loss)
            where delta_w = w_local - w_global
        """
        self.set_parameters(global_params)
        self.model.train()

        # Keep a tensor copy of global parameters on device for FedProx penalty
        global_tensors = {
            k: torch.tensor(v, device=self.device, dtype=torch.float32)
            for k, v in global_params.items()
        }

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

                # FedProx proximal regularization: (mu / 2) * ||w - w_t||^2
                if mu_fedprox > 0.0:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_tensors:
                            prox_loss += torch.sum((param - global_tensors[name]) ** 2)
                    loss += (mu_fedprox / 2.0) * prox_loss

                loss.backward()
                optimizer.step()

                batch_size = len(batch_y)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        # Compute parameter update delta: w_local - w_global
        local_params = self.get_parameters()
        delta: Dict[str, np.ndarray] = {
            k: (local_params[k] - global_params[k]).astype(np.float32)
            for k in global_params.keys()
        }

        # DP post-processing (shared, unit-tested numpy helpers; see client/dp.py)
        if clip_norm is not None and clip_norm > 0.0:
            delta, _ = clip_update(delta, clip_norm)
        if noise_multiplier > 0.0:
            if clip_norm is None or clip_norm <= 0.0:
                raise ValueError("noise requires a clip_norm: sensitivity is unbounded otherwise")
            delta = add_gaussian_noise(delta, per_client_noise_std(noise_multiplier, clip_norm, honest_min))

        avg_loss = total_loss / max(1, total_samples)
        return delta, len(self.train_loader.dataset), avg_loss

    def evaluate(self) -> Dict[str, float]:
        """Evaluates current local model on the client's validation set."""
        self.model.eval()
        all_probs: list[float] = []
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.numpy().tolist())

        y_true = np.array(all_labels)
        y_scores = np.array(all_probs)
        y_pred = np.array(all_preds)

        return compute_threat_metrics(y_true, y_scores, y_pred, target_fpr=0.001)
