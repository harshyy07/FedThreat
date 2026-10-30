"""Feature extractor producing fixed-length vectors from CanonicalEvents."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

from fedthreat.features.spec import CanonicalEvent, FeatureSpec
from fedthreat.features.hashing import simple_hash


class FeatureExtractor:
    """Transforms lists of canonical events into fixed-length normalized feature vectors."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    @property
    def output_dim(self) -> int:
        """Computes total dimensionality of extracted feature vector."""
        # 1. Numerical features
        dim = len(self.spec.numerical_features)
        
        # 2. Hashed features
        for h in self.spec.hashed_features:
            dim += int(h.get("num_bins", 16))
            
        # 3. Categorical one-hot features (basic protocol/action slots: tcp, udp, icmp, allow, drop, block)
        # 6 fixed categorical slots for v1
        dim += 6
        
        # 4. If mask vector included, we concatenate a mask indicator of equal dimension
        if self.spec.include_mask_vector:
            dim *= 2
            
        return dim

    def extract_from_event(self, event: CanonicalEvent) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts numerical & hashed features along with a missingness mask.
        
        Returns:
            (combined_vector, feature_values)
        """
        vals: List[float] = []
        masks: List[float] = []

        # 1. Numerical features
        for num_feat in self.spec.numerical_features:
            if hasattr(event, num_feat):
                val = getattr(event, num_feat)
                if val is not None:
                    vals.append(float(val))
                    masks.append(1.0)
                else:
                    vals.append(0.0)
                    masks.append(0.0)
            elif num_feat in event.extra_fields:
                val = event.extra_fields[num_feat]
                if val is not None:
                    vals.append(float(val))
                    masks.append(1.0)
                else:
                    vals.append(0.0)
                    masks.append(0.0)
            else:
                vals.append(0.0)
                masks.append(0.0)

        # 2. Hashed features
        for h in self.spec.hashed_features:
            src = h.get("source_field", "")
            nbins = int(h.get("num_bins", 16))
            raw_val = getattr(event, src, None) or event.extra_fields.get(src, None)
            if raw_val is not None:
                h_vec = simple_hash(str(raw_val), nbins)
                vals.extend(h_vec.tolist())
                masks.extend([1.0] * nbins)
            else:
                vals.extend([0.0] * nbins)
                masks.extend([0.0] * nbins)

        # 3. Standard categorical mappings
        proto_map = {"tcp": 0, "udp": 1, "icmp": 2}
        proto_vec = [0.0, 0.0, 0.0]
        if event.protocol.lower() in proto_map:
            proto_vec[proto_map[event.protocol.lower()]] = 1.0
            proto_mask = [1.0, 1.0, 1.0]
        else:
            proto_mask = [0.0, 0.0, 0.0]
        vals.extend(proto_vec)
        masks.extend(proto_mask)

        action_map = {"allow": 0, "drop": 1, "block": 2}
        action_vec = [0.0, 0.0, 0.0]
        if event.action.lower() in action_map:
            action_vec[action_map[event.action.lower()]] = 1.0
            action_mask = [1.0, 1.0, 1.0]
        else:
            action_mask = [0.0, 0.0, 0.0]
        vals.extend(action_vec)
        masks.extend(action_mask)

        feature_arr = np.array(vals, dtype=np.float32)
        mask_arr = np.array(masks, dtype=np.float32)

        if self.spec.include_mask_vector:
            full_vector = np.concatenate([feature_arr, mask_arr])
        else:
            full_vector = feature_arr

        return full_vector, feature_arr

    def extract_batch(self, events: List[CanonicalEvent]) -> np.ndarray:
        """Extracts batch of canonical events into a 2D numpy array (N, D)."""
        if not events:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        vecs = [self.extract_from_event(e)[0] for e in events]
        return np.vstack(vecs)
