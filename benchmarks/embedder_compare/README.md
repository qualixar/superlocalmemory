# Embedder comparison harness

Compares embedders on one shared corpus (LoCoMo conversation turns, screenshots,
synthetic images, PDF pages) at the embedder level: raw cosine retrieval, no daemon.

| System | Text | Images and PDF pages |
|---|---|---|
| `bm25` | BM25 (model-free smoke test, never used for a verdict) | BM25 over OCR text |
| `s1` | nomic-embed-text-v1.5 exactly as SLM ships it (raw text, no task prefixes) | OCR text (RapidOCR) embedded by nomic |
| `s1p` | optional, reference only: nomic with its recommended task prefixes | same as `s1` |
| `s2` | EmbeddingGemma 2 text-only loadout | EmbeddingGemma 2 full model, same vector space |
| `s3` | nomic-embed-text-v1.5 | EmbeddingGemma 2 media channel, fused with nomic by weighted reciprocal-rank fusion (k=60); the media weight (0, 0.1, 0.25, 0.5, 0.75, 1.0) is chosen on dev |

Every model loadout runs in its own process, so load time and peak RSS belong to one loadout.
The EmbeddingGemma 2 full loadout (images and media queries) runs once and is reused by `s3` after `s2`.

## Run

Run everything from this folder (`cd benchmarks/embedder_compare`); the commands below use relative paths.

```bash
export SLM_BENCH_HOME=~/.cache/slm-embedder-compare     # generated data lives here, never in the repo
uv venv ~/venvs/bench -p 3.11 && uv pip install -p ~/venvs/bench -r requirements-harness.txt
./fetch_data.sh                                          # LoCoMo at a pinned commit + sha256 check
~/venvs/bench/bin/python build_dataset.py                # corpus, queries, qrels, manifest.lock
                                                         # (add --refreeze to accept a changed test set)
~/venvs/bench/bin/python run_eval.py --systems bm25      # smoke run, no model needed
~/venvs/bench/bin/python run_eval.py --systems s1,s2,s3 \
    --python-slm /path/to/slm-venv/bin/python --python-eg2 /path/to/eg2-venv/bin/python
~/venvs/bench/bin/python report.py --out results/        # RESULTS.md + RESULTS.json
```

Model venvs: `slm` uses SLM's pins (`sentence-transformers==5.6.1`, `transformers==5.10.4`,
torch from `pyproject.toml`); `eg2` needs `transformers>=5.19`, `sentence-transformers>=6.1`
and `torchvision` (never the audio extra). CPU only. EmbeddingGemma 2 is loaded with
`attn_implementation="sdpa"` and float32; an earlier measurement on this class of CPU saw about 46 s per image idle with default kwargs and
1,094 ms with these settings (this harness does not re-measure that). The audio tower is never loaded.
A system whose venv is missing or fails is listed as "not run (reason)" in the report.

Tests (offline, no models): `pytest benchmarks/embedder_compare/tests -p no:cacheprovider -o addopts=""`.

## Query set (136 queries, all labels PROVISIONAL: maintainer review pending)

text single-hop 20, multi-hop 14, entity 14, temporal 18 (LoCoMo, 3 conversations), image 30,
PDF page 18, unanswerable 22 (12 LoCoMo adversarial, 10 about content never stored).
Split dev 40% / test 60% per stratum, assigned by sha256 of the query id.
`golden/test_manifest.sha256` is the committed hash of the test queries and qrels. The build
refuses to change it unless run with `--refreeze`; `run_eval.py` records a hash of the whole
dataset in each run; `report.py` refuses a changed test set, a run made on another dataset build,
or a run that lacks any query. Groups with n < 30 are flagged: no stratum-level claim.

Most image and PDF queries target text-bearing pictures (screenshots, slides, charts, dialogs),
which OCR handles well, so the media strata mostly measure OCR plus text embedding. Only six
synthetic images (`vis_*`) are purely visual (shapes and colours, no text); their queries share
no words with anything printed in an image. Do not read the media strata as a general
image-understanding score.

`golden/locomo_selection.json` holds ids only. `golden/media_queries.jsonl` holds hand-written
queries for images and PDF pages; an `anchor` phrase is checked at build time to appear on
exactly the labelled page.

## Metrics

Recall@5 (primary), MRR@10, nDCG@10 on answerable queries; false-answer rate and abstention
precision/recall on unanswerable ones (abstain when the top-1 score is below a threshold tuned
on dev only), reported separately for LoCoMo adversarial questions (stored but misattributed) and
never-stored media queries; for `s3` the abstention score is the raw cosine of the fused top-1
document in the channel it came from; 95% paired bootstrap CIs (10,000 resamples, seed 1729); Fisher randomisation test
via ranx; latency p50/p95 per query and per item, load time, peak RSS per model process (100 ms
sampling), items per minute. One-model rule: pass when the lower bound of the 95% CI of the
paired text recall@5 delta (S2 minus S1, S1 being nomic as shipped) is at least -0.03. This is
"criterion 1 of 4 (text non-inferiority)". Outcomes: PASS, NOT SHOWN NON-INFERIOR (point delta at
least -0.03 but CI lower bound below it), FAIL (point delta below -0.03), SAMPLE TOO SMALL TO
DECIDE (below 30 text test queries) and INVALID (S1 text recall@5 is 0 or below bm25). The CI width
is printed. Timing runs do one untimed warm-up call per kind first; OCR time per media item is
reported separately from embedding time.

## Data and licences

- LoCoMo (snap-research/locomo, CC BY-NC 4.0, commit 3eb6f2c): downloaded at run time, never
  committed. Only ids are stored here.
- Screenshots come from `docs/screenshots/`; synthetic images are generated by `synth_images.py`;
  PDFs are generated from this repo's public docs by `synth_pdfs.py` (plus one image-only PDF).
  `fetch_data.sh` also tries the author's public arXiv paper; the build does not use it yet.
- The root `.gitignore` ignores `/benchmarks/`, so new files here need `git add -f <path>`
  (never add `__pycache__` or `*.pyc`).
- Privacy: put private images or PDFs only under `private/`. It is git-ignored and must never be
  uploaded; real-user data stays on the machine that owns it.

## End-to-end daemon test

`e2e_daemon.py` starts a real SLM daemon from a given install, ingests the LoCoMo turns through
`POST /remember`, runs the text queries through `GET /recall`, and writes `e2e.json` and `E2E.md`.
It records startup and ready times, latencies, recall@5 and MRR@10 per group (test and dev),
unanswerable outcomes, answer-check fields, channel status, peak RSS per process and log errors.
It never prints the daemon capability token.

```bash
~/venvs/bench/bin/python e2e_daemon.py --slm-bin ~/venvs/slm/bin/slm --offline --out results/e2e/
```
