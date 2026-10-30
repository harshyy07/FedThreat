"""Prints the feature-spec hash and writes initial server params next to the server certs."""
import sys
from pathlib import Path

import numpy as np

from fedthreat.client.numpy_trainer import init_linear_params
from fedthreat.features.default import default_spec
from fedthreat.features.extractor import FeatureExtractor

spec = default_spec()
if len(sys.argv) > 1:
    np.savez(Path(sys.argv[1]) / "server" / "initial_params.npz", **init_linear_params(FeatureExtractor(spec).output_dim))
print(spec.compute_spec_hash())
