"""Canonical event and feature specification models for FedThreat."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CanonicalEvent(BaseModel):
    """Normalized security log event schema based on ECS / OCSF standards."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    src_ip: Optional[str] = None
    dst_ip: Optional[str] = None
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    protocol: str = "tcp"
    action: str = "allow"
    service: str = "unknown"
    duration_ms: float = 0.0
    src_bytes: float = 0.0
    dst_bytes: float = 0.0
    total_packets: float = 0.0
    failed_auth_count: float = 0.0
    label: Optional[str] = None  # Attack label or "benign"
    extra_fields: Dict[str, Any] = Field(default_factory=dict)


class FeatureSpec(BaseModel):
    """Specification of feature extraction and hashing configuration."""
    version: int = 1
    spec_name: str = "network_flow_v1"
    window_minutes: int = 5
    numerical_features: List[str] = Field(default_factory=list)
    categorical_features: List[str] = Field(default_factory=list)
    hashed_features: List[Dict[str, Any]] = Field(default_factory=list)
    include_mask_vector: bool = True

    def compute_spec_hash(self) -> str:
        """Computes a deterministic SHA256 hash of the feature spec for round verification."""
        dumped = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
