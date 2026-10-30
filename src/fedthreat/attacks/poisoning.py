"""Byzantine and poisoning attack client implementations for FedThreat."""

from typing import Any, Dict
import numpy as np
from fedthreat.base import Attacker


class SignFlippingAttacker(Attacker):
    """Sign-flipping / negative scaling attack.
    
    Inverts the update direction and scales it to steer global model away from convergence.
    """

    def __init__(self, scale: float = -2.0):
        self.scale = scale

    def poison_update(
        self, clean_update: Dict[str, np.ndarray], round_num: int, context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        poisoned = {}
        for k, v in clean_update.items():
            poisoned[k] = (v * self.scale).astype(np.float32)
        return poisoned


class ModelReplacementAttacker(Attacker):
    """Model-replacement / update-scaling attack (Bagdasaryan et al., 2020).
    
    Scales a targeted malicious update by n / eta so that after FedAvg averaging across
    all n participants, the malicious update dominates and overrides the global model.
    """

    def __init__(self, target_boost: float = 5.0):
        self.target_boost = target_boost

    def poison_update(
        self, clean_update: Dict[str, np.ndarray], round_num: int, context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        num_clients = float(context.get("num_clients", 5))
        scale = self.target_boost * num_clients
        poisoned = {}
        for k, v in clean_update.items():
            # Invert sign and boost magnitude
            poisoned[k] = (-v * scale).astype(np.float32)
        return poisoned


class GaussianNoiseAttacker(Attacker):
    """Blatant Byzantine attacker injecting high-variance Gaussian noise to corrupt model weights."""

    def __init__(self, noise_std: float = 10.0):
        self.noise_std = noise_std

    def poison_update(
        self, clean_update: Dict[str, np.ndarray], round_num: int, context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        poisoned = {}
        for k, v in clean_update.items():
            noise = np.random.normal(0.0, self.noise_std, size=v.shape).astype(np.float32)
            poisoned[k] = noise
        return poisoned
