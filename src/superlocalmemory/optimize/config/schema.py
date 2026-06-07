"""Typed schema for optimize.json.

All fields have a default — partial JSON is always valid (additive migration).
No imports from optimize.storage (no circular dependency).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TTLConfig:
    """TTL settings for each cache tier, in seconds."""

    exact_seconds: int = 86400
    semantic_seconds: int = 3600
    ccr_seconds: int = 604800
    sweep_interval_seconds: int = 3600

    def validate(self) -> None:
        for fname in ("exact_seconds", "semantic_seconds", "ccr_seconds",
                      "sweep_interval_seconds"):
            v = getattr(self, fname)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(
                    f"TTLConfig.{fname} must be a positive integer, got {v!r}"
                )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TTLConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ProviderConfig:
    """Per-provider settings."""

    enabled: bool = True
    base_url: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProviderConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            base_url=str(d.get("base_url", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "base_url": self.base_url}


@dataclass(frozen=True)
class OptimizeConfig:
    """Master typed schema for ~/.superlocalmemory/optimize.json.

    INTERFACE-CONTRACT §2 CONFORMANCE — all fields frozen, no aliases.
    """

    # Master kill-switch
    enabled: bool = False
    proxy_enabled: bool = False

    # Cache
    cache_enabled: bool = True
    semantic_enabled: bool = False
    semantic_return_threshold: float = 0.98
    semantic_verify_lo: float = 0.90
    semantic_error_target: float = 0.02
    semantic_explore_rate: float = 0.10
    semantic_constant_time: bool = False
    semantic_centroid_defense: bool = True
    semantic_multiturn_guard: bool = True
    semantic_ann_top_k: int = 5
    semantic_boundary_init: float = 0.95
    semantic_boundary_floor: float = 0.85
    semantic_boundary_ceiling: float = 0.995
    semantic_boundary_step: float = 0.01
    semantic_max_turns_for_semantic: int = 6
    semantic_context_window_turns: int = 3
    semantic_centroid_distance_floor: float = 0.15
    semantic_verifier_model: str = ""
    semantic_pad_latency_ms: float = 0.0
    semantic_centroid_min_similarity: float = 0.85

    # Compress
    compress_enabled: bool = True
    compress_mode: str = "safe"
    compress_code: bool = True
    compress_json: bool = True
    compress_prose: bool = False
    compress_ccr: bool = True
    compress_align: bool = True
    compress_protect_recent: int = 4
    compress_llmlingua_allow_download: bool = False

    # TTL + providers + pricing
    ttl_seconds: int = 86400
    providers: dict = field(default_factory=dict)
    pricing: dict = field(default_factory=dict)

    # Metrics/savings
    active_model: str | None = None
    usd_to_inr_rate: float = 83.5
    pricing_overrides: dict = field(default_factory=dict)

    # Observability
    prometheus_port: int = 9091

    # TTL sub-config
    ttl: "TTLConfig" = field(default_factory=lambda: TTLConfig())

    # Config version
    config_version: int = 1

    def validate(self) -> None:
        if self.compress_mode not in ("safe", "aggressive"):
            raise ValueError(
                f"compress_mode must be 'safe' or 'aggressive', "
                f"got {self.compress_mode!r}"
            )
        if not (0.0 < self.semantic_return_threshold <= 1.0):
            raise ValueError(
                f"semantic_return_threshold must be in (0.0, 1.0], "
                f"got {self.semantic_return_threshold!r}"
            )
        if not (0.0 < self.semantic_error_target < 1.0):
            raise ValueError(
                f"semantic_error_target must be in (0.0, 1.0), "
                f"got {self.semantic_error_target!r}"
            )
        if self.semantic_pad_latency_ms > 100:
            logger.warning(
                "semantic_pad_latency_ms=%s may negate cache latency benefit",
                self.semantic_pad_latency_ms,
            )
        if self.prometheus_port < 1024 or self.prometheus_port > 65535:
            raise ValueError(
                f"prometheus_port must be 1024-65535, got {self.prometheus_port!r}"
            )
        self.ttl.validate()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizeConfig":
        ttl_raw = d.get("ttl", {})
        ttl = TTLConfig.from_dict(ttl_raw) if isinstance(ttl_raw, dict) else TTLConfig()

        providers_raw = d.get("providers", {})
        providers: dict[str, ProviderConfig] = {}
        if isinstance(providers_raw, dict):
            for pname, pdata in providers_raw.items():
                if isinstance(pdata, dict):
                    providers[pname] = ProviderConfig.from_dict(pdata)

        return cls(
            enabled=bool(d.get("enabled", False)),
            proxy_enabled=bool(d.get("proxy_enabled", False)),
            cache_enabled=bool(d.get("cache_enabled", True)),
            semantic_enabled=bool(d.get("semantic_enabled", False)),
            semantic_return_threshold=float(d.get("semantic_return_threshold", 0.98)),
            semantic_verify_lo=float(d.get("semantic_verify_lo", 0.90)),
            semantic_error_target=float(d.get("semantic_error_target", 0.02)),
            semantic_explore_rate=float(d.get("semantic_explore_rate", 0.10)),
            semantic_constant_time=bool(d.get("semantic_constant_time", False)),
            semantic_centroid_defense=bool(d.get("semantic_centroid_defense", True)),
            semantic_multiturn_guard=bool(d.get("semantic_multiturn_guard", True)),
            semantic_ann_top_k=int(d.get("semantic_ann_top_k", 5)),
            semantic_boundary_init=float(d.get("semantic_boundary_init", 0.95)),
            semantic_boundary_floor=float(d.get("semantic_boundary_floor", 0.85)),
            semantic_boundary_ceiling=float(d.get("semantic_boundary_ceiling", 0.995)),
            semantic_boundary_step=float(d.get("semantic_boundary_step", 0.01)),
            semantic_max_turns_for_semantic=int(d.get("semantic_max_turns_for_semantic", 6)),
            semantic_context_window_turns=int(d.get("semantic_context_window_turns", 3)),
            semantic_centroid_distance_floor=float(d.get("semantic_centroid_distance_floor", 0.15)),
            semantic_verifier_model=str(d.get("semantic_verifier_model", "")),
            semantic_pad_latency_ms=float(d.get("semantic_pad_latency_ms", 0.0)),
            semantic_centroid_min_similarity=float(
                d.get("semantic_centroid_min_similarity", 0.85)
            ),
            compress_enabled=bool(d.get("compress_enabled", True)),
            compress_mode=str(d.get("compress_mode", "safe")),
            compress_code=bool(d.get("compress_code", True)),
            compress_json=bool(d.get("compress_json", True)),
            compress_prose=bool(d.get("compress_prose", False)),
            compress_ccr=bool(d.get("compress_ccr", True)),
            compress_align=bool(d.get("compress_align", True)),
            compress_protect_recent=int(d.get("compress_protect_recent", 4)),
            compress_llmlingua_allow_download=bool(
                d.get("compress_llmlingua_allow_download", False)
            ),
            ttl_seconds=int(d.get("ttl_seconds", 86400)),
            providers=providers,
            pricing=dict(d.get("pricing", {})),
            active_model=str(d.get("active_model", None)) if d.get("active_model") else None,
            usd_to_inr_rate=float(d.get("usd_to_inr_rate", 83.5)),
            pricing_overrides=dict(d.get("pricing_overrides", {})),
            prometheus_port=int(d.get("prometheus_port", 9091)),
            ttl=ttl,
            config_version=int(d.get("config_version", 1)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "proxy_enabled": self.proxy_enabled,
            "cache_enabled": self.cache_enabled,
            "semantic_enabled": self.semantic_enabled,
            "semantic_return_threshold": self.semantic_return_threshold,
            "semantic_verify_lo": self.semantic_verify_lo,
            "semantic_error_target": self.semantic_error_target,
            "semantic_explore_rate": self.semantic_explore_rate,
            "semantic_constant_time": self.semantic_constant_time,
            "semantic_centroid_defense": self.semantic_centroid_defense,
            "semantic_multiturn_guard": self.semantic_multiturn_guard,
            "semantic_ann_top_k": self.semantic_ann_top_k,
            "semantic_boundary_init": self.semantic_boundary_init,
            "semantic_boundary_floor": self.semantic_boundary_floor,
            "semantic_boundary_ceiling": self.semantic_boundary_ceiling,
            "semantic_boundary_step": self.semantic_boundary_step,
            "semantic_max_turns_for_semantic": self.semantic_max_turns_for_semantic,
            "semantic_context_window_turns": self.semantic_context_window_turns,
            "semantic_centroid_distance_floor": self.semantic_centroid_distance_floor,
            "semantic_verifier_model": self.semantic_verifier_model,
            "semantic_pad_latency_ms": self.semantic_pad_latency_ms,
            "semantic_centroid_min_similarity": self.semantic_centroid_min_similarity,
            "compress_enabled": self.compress_enabled,
            "compress_mode": self.compress_mode,
            "compress_code": self.compress_code,
            "compress_json": self.compress_json,
            "compress_prose": self.compress_prose,
            "compress_ccr": self.compress_ccr,
            "compress_align": self.compress_align,
            "compress_protect_recent": self.compress_protect_recent,
            "compress_llmlingua_allow_download": self.compress_llmlingua_allow_download,
            "ttl_seconds": self.ttl_seconds,
            "providers": {k: v.as_dict() for k, v in self.providers.items()},
            "pricing": self.pricing,
            "active_model": self.active_model,
            "usd_to_inr_rate": self.usd_to_inr_rate,
            "pricing_overrides": self.pricing_overrides,
            "prometheus_port": self.prometheus_port,
            "ttl": {
                "exact_seconds": self.ttl.exact_seconds,
                "semantic_seconds": self.ttl.semantic_seconds,
                "ccr_seconds": self.ttl.ccr_seconds,
                "sweep_interval_seconds": self.ttl.sweep_interval_seconds,
            },
            "config_version": self.config_version,
        }

