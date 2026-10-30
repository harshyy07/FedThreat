"""Implementation of standard FedAvg and FedProx server aggregation."""

from typing import Dict, List
import numpy as np
from fedthreat.base import Aggregator


class FedAvgAggregator(Aggregator):
    """Federated Averaging aggregator (McMahan et al.).
    
    Computes sample-weighted average of parameter deltas or weights.
    """

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Aggregates client update deltas into new global parameters."""
        if not client_updates:
            return global_params

        total_weight = sum(client_weights)
        if total_weight == 0:
            normalized_weights = [1.0 / len(client_weights)] * len(client_weights)
        else:
            normalized_weights = [w / total_weight for w in client_weights]

        new_params = {}
        for key in global_params.keys():
            # Weighted sum of deltas
            agg_delta = np.zeros_like(global_params[key], dtype=np.float32)
            for client_delta, weight in zip(client_updates, normalized_weights):
                agg_delta += weight * client_delta[key]
            new_params[key] = global_params[key] + agg_delta

        return new_params
