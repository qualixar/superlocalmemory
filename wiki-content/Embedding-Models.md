# Embedding Models

SLM uses two local embedding models, each for one job. Text is always searched with `nomic-ai/nomic-embed-text-v1.5`. If you turn on [Pictures and Documents](Pictures-and-Documents), a second model, `google/embeddinggemma-2` (EmbeddingGemma 2), reads pictures and PDF pages, and one recall returns text and pictures together. The two never replace each other: your text vectors are not touched when you turn pictures on, and the picture model is not used for text.

## Why text stays on its own model

Recall decides "this is in your memory" with an answer check that is fitted to one text model. Vectors from different models also cannot be compared, so moving text to another model means re-reading every memory. Moving text to EmbeddingGemma 2 was tried and measured on the release harness, a synthetic store: with the check fitted for the text model, EmbeddingGemma 2 made recall refuse all 203 real questions. A check fitted for EmbeddingGemma 2 found every real answer on questions it had not seen, but it also answered questions that have no answer: it refused 69% of made-up questions where the text model refuses 98.5%. So that check is not used, and text stays on the text model.

For that reason "Upgrade memory engine", the card in Settings and `slm embedder upgrade` that used to move text to EmbeddingGemma 2, is no longer offered. The command name remains, but `slm embedder --help` now describes it as not offered, and it does not start anything.

## What the picture model costs

| Fact | Value |
|---|---|
| Used for | Pictures and PDF pages only, never text |
| Download the first time | About 1.5 GB, with the OCR and PDF tools ([Pictures and Documents](Pictures-and-Documents)) |
| Computer | Sets up by itself from 16 GB of memory (machines sold as 16 GB count); below that it is one click, with a warning |
| When idle | Unloads after 30 minutes without use (`SLM_MEDIA_WORKER_IDLE_S` changes it), so a text-only day holds no picture model in memory |

A recall never waits for the picture model to load. If it is cold, that recall goes on without pictures and says so, and the model warms in the background ([How Recall Works](How-Recall-Works)).

## Switching the text model yourself

You can still choose a different text model, for example a hosted one with your own key or a model served by Ollama. A switch re-reads every memory in the background inside the running service. Recall and remember keep using the current model until the new vectors are complete, and then both change in one step.

```bash
slm embedder switch nomic-ai/nomic-embed-text-v1.5
slm embedder switch text-embedding-3-small --provider openai --dimension 1536
slm embedder status            # progress, the model in use, and whether a previous one is kept
slm embedder cancel            # stop a running switch; the current model stays
slm embedder rollback          # go back to the previous model
slm embedder forget-previous   # free the previous model's vectors (ends rollback)
```

`switch` also takes `--endpoint` for an OpenAI-compatible URL, `--task-prefixes` for the model's retrieval task prompts, `--no-wait` to return once queued, and `--json`. `--provider` is `sentence-transformers`, `ollama`, `openai` or `slm-media`; the default is the current one. The dimension must be 64 to 8192 and must be given for a new model. A hosted model's key is never taken on the command line; set it in the dashboard or in `config.json`. These commands talk to the running service, so start it with `slm serve` if it is not running.

The same switch is available in the dashboard under Settings, in the Embeddings & Retrieval group ([Configuration](Configuration)).

### The refusal and `--force`

A model that SLM manages itself (provider `slm-media`, which is how EmbeddingGemma 2 is served) is refused as a text model unless the answer check is fitted for it. The refusal applies from every route: the Settings card, the embedding picker, the config file, the command line and MCP. Only this command moves text anyway:

```bash
slm embedder switch google/embeddinggemma-2 --provider slm-media --dimension 768 --force
```

Use `--force` only if you accept that recall may say a memory is not there when it is. A model you configure yourself (OpenAI, Ollama) is your choice and is not refused.

## If your text was moved by an older release

Earlier releases offered "Upgrade memory engine" and moved some stores' text to EmbeddingGemma 2. If yours was moved, three places tell you:

- The Settings card appears, and shows that text runs on a model the answer check is not fitted for.
- `slm doctor` has a check named "Answer check fit".
- The MCP `health` tool reports the same state.

Going back takes minutes, because the previous vectors were kept: press **Roll back** on the card, or run `slm embedder rollback`. Your memories are not changed either way. **Free the old data** (`slm embedder forget-previous`) deletes the previous vectors and ends the option to roll back, so use it only once you are sure.

## What happens during a switch

A job moves through the states `queued`, `running`, `catching_up`, `ready` and `activated`; the other end states are `failed`, `rolled_back` and `cancelled`.

1. The service loads the new model and asks it one question to prove it is loaded.
2. It embeds every memory into a separate staging space, in batches of 32 with short rests, so recall and remember keep using the old model and old vectors.
3. It catches up with memories written or edited meanwhile, using a change log, so nothing saved during the switch is missed.
4. It waits for a quiet moment, pauses background work, drains the requests already admitted and holds new ones for an instant. In that window one transaction makes the new space live, `config.json` is rewritten and the new engine is published. Requests resume only after that, so no request pairs the new model with the old vectors.

If anything fails, the old space stays live. A restart resumes a running job at its cursor. SLM keeps exactly one previous space. The next switch moves the then-live space into the previous slot and drops the older one, so the option to roll back lasts only as long as you do not switch again. The tuning variables are `SLM_REINDEX_BATCH` (default 32), `SLM_REINDEX_PAUSE_S` (0.02) and `SLM_REINDEX_DUTY_REST` (1.0).

The HTTP routes are under `/api/v3/embedding/reindex`: `GET` for status, `POST` to start a switch, `/rollback`, `/cancel` and `/forget-previous`. Changes need MANAGE permission.

## Calibrating the answer check on your store

`slm embedder calibrate` measures the answer check and the embedding model's evidence floor on your own memories, in the background; `calibration-status` shows progress and `calibration-cancel` stops it. It is the command-line twin of **Calibrate on my store** in Settings, Answer check ([Answer Check](Answer-Check)). A new calibration is rejected if it does worse than the current one on held-out questions.

Code: `core/support_fit_guard.py`, `retrieval/support_recipe.py`, `core/embedding_reindex.py`, `core/embedding_reindex_steps.py`, `core/embedding_reindex_activate.py`, `storage/embedding_spaces.py`, `storage/embedding_space_swap.py`, `cli/embedder_cmd.py`, `cli/doctor_models.py`, `server/routes/embedding_reindex.py`, `runtimes/media_models.py`, `ui/js/od-engine-upgrade.js`.
