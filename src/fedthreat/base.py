"""Abstract base classes for FedThreat components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np

from fedthreat.features.spec import CanonicalEvent


class LogAdapter(ABC):
    """Abstract base class for converting organization-specific raw logs to canonical events."""

    @abstractmethod
    def parse_line(self, line: str) -> CanonicalEvent:
        """Parses a single log entry into a CanonicalEvent."""
        pass

    @abstractmethod
    def parse_file(self, filepath: str) -> List[CanonicalEvent]:
        """Parses a batch or file of logs into CanonicalEvents."""
        pass


class Aggregator(ABC):
    """Abstract base class for server-side aggregation strategies (FedAvg, robust, etc.)."""

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_weights: List[float],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Aggregates a list of client parameter updates into a new global parameter set."""
        pass


class Attacker(ABC):
    """Abstract base class for simulating malicious client attacks."""

    @abstractmethod
    def poison_update(
        self,
        clean_update: Dict[str, np.ndarray],
        round_num: int,
        context: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Modifies a benign client update to perform poisoning, model replacement, or backdoors."""
        pass
