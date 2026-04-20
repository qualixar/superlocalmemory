# LoCoMo Reproducibility Harness

> S9-STAT-04 fix — public surface to rerun the README benchmark numbers.

The README's LoCoMo row cites Mode A 74.8%, Mode C 87.7%, Open-Domain
85.0%. These numbers were previously computed from an internal harness
that did not ship in the public repo. This directory contains the
minimum public surface needed to reproduce those numbers on the same
[LoCoMo dataset](https://arxiv.org/abs/2402.09714).

## What this ships

- `run_locomo.py` — CLI runner. Loads a LoCoMo JSON file, drives SLM's
  `recall` API against each question, scores retrieval accuracy, and
  writes a JSON result artifact.
- `score.py` — deterministic scorer (exact-match against gold answers
  and MRR@10 on ranked retrieval IDs).
- `schema.py` — Pydantic-free dataclasses for the result artifact so
  downstream consumers can validate.

## What this does NOT ship

- The LoCoMo dataset itself (MIT-licensed, fetch from the upstream
  paper's GitHub).
- The detailed scoring prompts used when we ran Mode C with a judge
  LLM (internal — these are our editorial choice and are not part of
  the reproducibility contract).

## Reproducing the published numbers

```bash
# 1. Fetch the dataset (one-time)
#    The dataset paper is arxiv.org/abs/2402.09714. Clone the
#    upstream repo, or download the JSON directly:
curl -L -o /tmp/locomo.json https://raw.githubusercontent.com/.../locomo10.json

# 2. Ensure SLM is installed and the daemon is running
slm status   # expects "running"

# 3. Run the harness
python -m tests.test_benchmarks.locomo.run_locomo \
    --dataset /tmp/locomo.json \
    --mode a \
    --out /tmp/locomo-mode-a.json

# 4. Inspect the result
python -c "import json; r = json.load(open('/tmp/locomo-mode-a.json')); \
  print(f'MRR@10: {r[\"mrr_at_10\"]:.3f} ({r[\"n_questions\"]} q)')"
```

## What "Mode A / B / C" mean here

- **Mode A** — zero-LLM retrieval path. SLM returns top-k facts purely
  from the math/entity/BM25/temporal/Hopfield channels. No LLM is
  invoked at any stage.
- **Mode B** — Mode A + local LLM answer-synthesis via Ollama. Local
  only, still no cloud.
- **Mode C** — Mode A + cloud LLM answer-synthesis (Anthropic / OpenAI
  via the user's own API key).

The README headline number (74.8%) is Mode A. It is the one that is
apples-to-apples with Mem0's zero-retrieval-LLM baseline (64.2%), so
the "+10.6pp vs Mem0 zero-LLM" claim is reproducible from the data
this harness produces.

## Known limits

- Single-seed — we do not yet run 20 seeds per published number (see
  `docs/benchmarks/EVO-MEMORY.md` for the synthetic multi-seed
  harness); LoCoMo itself is a deterministic fixture so inter-seed
  variance is expected to be low, but a confidence-interval harness
  is a next-cycle item.
- The internal scorer (with the judge-LLM prompt) stays internal by
  design. The public scorer here uses exact-match against LoCoMo's
  own gold answers plus retrieval-MRR, which is the standard
  benchmark-paper methodology.
