# Local resource-lifecycle evidence

Generated: `2026-07-15T06:13:09.501856+00:00`
Source: `b5c96e4484db7e4c57e28800dad54e4c9ec618da`
Package: `3.6.23`

## Protocol

- Mode A shipped local engine with `deterministic-sha256-numpy-768`.
- Warm-up: 5 ingest+recall pairs.
- Measured: 50 ingests and 100 fixed-state retry ingests and 100 fixed-corpus recalls.
- Temporary SQLite namespace; no provider call or network dependency.

## Machine

`macOS-26.5.2-arm64-arm-64bit`, `arm64`, Python `3.12.13`, 15 logical CPUs, 24576.0 MiB system memory.

## Measured latency

| path | samples | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|
| canonical ingest | 50 | 96.028 ms | 140.961 ms | 35.626 ms | 196.818 ms |
| idempotent retry ingest | 100 | 1.666 ms | 5.568 ms | 1.156 ms | 8.058 ms |
| repeat recall | 100 | 86.05 ms | 88.949 ms | 82.932 ms | 99.588 ms |

No universal latency budget is applied. These distributions belong to the machine, corpus, source commit, and mock model boundary above.

## RSS evidence

| phase | baseline | final | peak | growth | tail span | tail slope / 100 calls |
|---|---:|---:|---:|---:|---:|---:|
| corpus build (expected index growth) | 82.094 MiB | 114.844 MiB | 114.844 MiB | 32.75 MiB | 24.641 MiB | 123.203 MiB |
| fixed-state idempotent ingest | 114.922 MiB | 114.812 MiB | 114.922 MiB | -0.109 MiB | 0.0 MiB | 0.0 MiB |
| fixed-corpus repeat recall | 121.094 MiB | 128.891 MiB | 128.891 MiB | 7.797 MiB | 0.188 MiB | 0.281 MiB |

Bounded-window conclusion: **no_unbounded_growth_detected**. This finite run cannot prove behavior at unlimited corpus size.

## Explicit exclusions

Mode B Ollama and Mode C cloud-provider end-to-end latency are external model, host, provider, and network measurements. They are not inferred from this local run. Retrieval quality is also outside this harness.
