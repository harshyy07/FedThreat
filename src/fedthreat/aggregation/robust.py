"""Byzantine robust aggregation strategies: Median, Trimmed Mean, and Krum."""

from typing import Dict, List
import numpy as np
from fedthreat.base import Aggregator


class CoordinateWiseMedianAggregator(Aggregator):
    """Coordinate-wise Median aggregator (Yin et al., 2018).
    
    Robust against arbitrary Byzantine updates as long as f < n / 2.
    """

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if not client_updates:
            return global_params

        new_params = {}
        for key in global_params.keys():
            # Stack updates along axis 0: shape (num_clients, ...)
            stacked = np.stack([up[key] for up in client_updates], axis=0)
            median_delta = np.median(stacked, axis=0).astype(np.float32)
            new_params[key] = global_params[key] + median_delta

        return new_params


class TrimmedMeanAggregator(Aggregator):
    """Coordinate-wise Trimmed Mean aggregator (Yin et al., 2018).
    
    Trims the smallest and largest beta fraction of client updates along each coordinate.
    """

    def __init__(self, beta: float = 0.2, min_trim: int = 0):
        self.beta = beta
        self.min_trim = min_trim

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if not client_updates:
            return global_params

        n = len(client_updates)
        k = max(self.min_trim, int(np.floor(self.beta * n)))  # clients trimmed from each tail

        new_params = {}
        for key in global_params.keys():
            stacked = np.stack([up[key] for up in client_updates], axis=0)
            # Sort along client axis
            sorted_arr = np.sort(stacked, axis=0)

            # Slice away the k lowest and k highest values
            if k > 0 and (n - 2 * k) > 0:
                trimmed = sorted_arr[k : n - k]
            else:
                trimmed = sorted_arr

            mean_delta = np.mean(trimmed, axis=0).astype(np.float32)
            new_params[key] = global_params[key] + mean_delta

        return new_params


class KrumAggregator(Aggregator):
    """Krum / Multi-Krum aggregator (Blanchard et al., 2017).
    
    Selects the client update that is closest to its n - f - 2 neighbors in Euclidean distance.
    Multi-Krum averages the top-m chosen updates.
    """

    def __init__(self, num_malicious: int = 1, m: int = 1):
        self.f = num_malicious
        self.m = m

    def _flatten_update(self, update: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([v.flatten() for v in update.values()]).astype(np.float32)

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        n = len(client_updates)
        if n == 0:
            return global_params
        if n <= 2 * self.f + 2:
            # Fallback if too few clients to strictly satisfy Krum condition n >= 2f + 3
            # Choose nearest neighbor among all
            num_neighbors = max(1, n - self.f - 1)
        else:
            num_neighbors = n - self.f - 2

        # Flatten updates to calculate pairwise Euclidean distances
        flat_vectors = [self._flatten_update(u) for u in client_updates]
        dist_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flat_vectors[i] - flat_vectors[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Compute Krum score for each client: sum of distances to closest num_neighbors
        scores = []
        for i in range(n):
            dists = np.sort(dist_matrix[i])
            # dists[0] is distance to self (0), pick closest num_neighbors
            score = np.sum(dists[1 : num_neighbors + 1])
            scores.append(score)

        # Select top-m candidates with lowest Krum scores
        selected_indices = np.argsort(scores)[: max(1, min(self.m, n))]

        new_params = {}
        for key in global_params.keys():
            chosen_deltas = [client_updates[idx][key] for idx in selected_indices]
            avg_delta = np.mean(chosen_deltas, axis=0).astype(np.float32)
            new_params[key] = global_params[key] + avg_delta

        return new_params
