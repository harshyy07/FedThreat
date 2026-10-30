import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fedthreat.models.mlp import ThreatMLP
from fedthreat.client.trainer import ClientTrainer
from fedthreat.aggregation.fedavg import FedAvgAggregator


def test_fedavg_aggregator():
    agg = FedAvgAggregator()
    global_p = {"w": np.array([1.0, 2.0], dtype=np.float32)}
    
    # 2 clients
    client1_delta = {"w": np.array([2.0, 2.0], dtype=np.float32)}
    client2_delta = {"w": np.array([4.0, 0.0], dtype=np.float32)}

    # Equal weights
    res = agg.aggregate([client1_delta, client2_delta], [10.0, 10.0], global_p)
    # Expected: w + 0.5*[2, 2] + 0.5*[4, 0] = [1, 2] + [3, 1] = [4, 3]
    np.testing.assert_allclose(res["w"], np.array([4.0, 3.0], dtype=np.float32))


def test_client_trainer_delta():
    model = ThreatMLP(input_dim=8, hidden_dims=[16], num_classes=2)
    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=5)

    trainer = ClientTrainer(client_id=0, model=model, train_loader=loader, val_loader=loader, lr=1e-2)
    init_params = trainer.get_parameters()

    delta, n_samples, loss = trainer.train_round(init_params, epochs=1, mu_fedprox=0.01)
    assert n_samples == 20
    assert "backbone.0.weight" in delta
    # Check that training actually modified weights
    norm_delta = np.linalg.norm(delta["backbone.0.weight"])
    assert norm_delta > 0.0
