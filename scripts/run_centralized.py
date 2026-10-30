"""Centralized training baseline script for threat detection model."""

import yaml
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from fedthreat.features.spec import FeatureSpec
from fedthreat.features.extractor import FeatureExtractor
from fedthreat.models.mlp import ThreatMLP
from fedthreat.metrics import compute_threat_metrics
from fedthreat.sim.partitioner import generate_synthetic_flow_dataset


def run_centralized_baseline():
    print("=" * 60)
    print("FedThreat: Training Centralized Baseline (Upper Bound)")
    print("=" * 60)

    # 1. Load Feature Spec
    config_path = Path("configs/feature_spec_v1.yaml")
    with open(config_path, "r") as f:
        spec_dict = yaml.safe_load(f)
    spec = FeatureSpec(**spec_dict)
    extractor = FeatureExtractor(spec)
    print(f"[+] Loaded Feature Spec: {spec.spec_name} (Input Dim: {extractor.output_dim})")
    print(f"[+] Spec Hash: {spec.compute_spec_hash()[:16]}...")

    # 2. Generate Synthetic Dataset
    print("[+] Generating synthetic network flow dataset (15,000 events)...")
    events = generate_synthetic_flow_dataset(num_samples=15000, seed=42)

    # 3. Extract Features and Target Labels (Binary: 0=benign, 1=attack)
    print("[+] Extracting features into fixed-length vectors with mask representation...")
    X = extractor.extract_batch(events)
    y = np.array([0 if e.label == "benign" else 1 for e in events], dtype=np.int64)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardize numerical features using train statistics
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 4. Initialize ThreatMLP Model
    model = ThreatMLP(input_dim=extractor.output_dim, hidden_dims=[128, 64, 32], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[+] Training ThreatMLP for 10 epochs on {len(X_train)} samples...")
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        avg_loss = total_loss / len(X_train)
        print(f"    Epoch {epoch:02d}/10 | Loss: {avg_loss:.4f}")

    # 5. Evaluate with Threat Detection Metrics
    print("[+] Evaluating on held-out test set...")
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32)
        logits = model(test_x)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.argmax(logits, dim=1).numpy()

    metrics = compute_threat_metrics(y_true=y_test, y_scores=probs, y_pred=preds, target_fpr=0.001)

    print("-" * 60)
    print("Centralized Baseline Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  * {k:25s}: {v:.4f}")
    print("-" * 60)
    print("[SUCCESS] Centralized Baseline Execution Complete!")


if __name__ == "__main__":
    run_centralized_baseline()
