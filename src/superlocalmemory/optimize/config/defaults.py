"""Default OptimizeConfig singleton — import-safe."""

from __future__ import annotations

from superlocalmemory.optimize.config.schema import (
    OptimizeConfig, ProviderConfig, TTLConfig,
)

DEFAULT_OPTIMIZE_CONFIG = OptimizeConfig(
    enabled=False,
    proxy_enabled=False,
    cache_enabled=True,
    semantic_enabled=False,
    semantic_return_threshold=0.98,
    semantic_verify_lo=0.90,
    semantic_error_target=0.02,
    semantic_explore_rate=0.10,
    semantic_constant_time=False,
    semantic_centroid_defense=True,
    semantic_multiturn_guard=True,
    semantic_ann_top_k=5,
    semantic_boundary_init=0.95,
    semantic_boundary_floor=0.85,
    semantic_pad_latency_ms=0.0,
    semantic_centroid_min_similarity=0.85,
    compress_enabled=False,
    compress_mode="safe",
    compress_prose=False,
    compress_protect_recent=4,
    ttl_seconds=86400,
    ttl=TTLConfig(),
    providers={
        "anthropic": ProviderConfig(enabled=True, base_url=""),
        "openai": ProviderConfig(enabled=True, base_url=""),
        "gemini": ProviderConfig(enabled=True, base_url=""),
    },
    pricing={},
    active_model="claude-sonnet-4-6",
    usd_to_inr_rate=83.5,
    pricing_overrides={},
    prometheus_port=9091,
    config_version=1,
)
