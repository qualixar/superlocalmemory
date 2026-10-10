# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The models SLM knows about, and which to recommend on this computer (4.1.22).

One place for the setup wizard, the dashboard and ``slm models`` to read
from, so the three never disagree. Pure data and pure functions: nothing here
touches the network, a model or a file. The list of installed Ollama models
comes from the caller (``GET /api/tags``), and the machine's memory size too.

Local language models are ranked by SLM's own extraction test, not by
general benchmarks: each model extracts facts from the same synthetic
sentences, and SLM's source check counts the facts that changed what was
said (an invented date or number, a lost "not", a reversed order). An entry
that SLM has not run through that test says so.

The catalogue is a recommendation, never a restriction: any installed Ollama
model, any OpenRouter model id and any custom endpoint can still be chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bumped whenever an entry's numbers or advice change.
CATALOG_VERSION = "2026-10-08"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    id: str
    role: str  # "llm" | "embedder"
    provider: str  # "ollama" | "sentence-transformers" | "openrouter" | "slm-media"
    label: str
    advice: str  # one line a person reads: what it is good for, what to expect
    size_gb: float | None = None  # download size (local models)
    min_ram_gb: int | None = None  # total memory below which it is not offered first
    dimension: int | None = None  # embedders; None = measured when chosen
    #: SLM's own extraction test (local LLMs): sources out of ``tested`` that
    #: produced at least one fact that changed what was said. None = not tested.
    damaged: int | None = None
    tested: int | None = None
    price: str = ""  # hosted models: "$in / $out per million tokens"
    recommended: bool = False


@dataclass(frozen=True, slots=True)
class Recommendation:
    entry: ModelEntry | None  # None: installed but not in the catalogue
    model_id: str
    installed: bool
    fits: bool
    reason: str


def _e(**kw) -> ModelEntry:
    return ModelEntry(**kw)


LOCAL_LLMS: tuple[ModelEntry, ...] = (
    _e(id="gemma3:4b", role="llm", provider="ollama", label="Gemma 3 4B",
       size_gb=3.3, min_ram_gb=8, damaged=0, tested=102, recommended=True,
       advice="Most faithful in SLM's extraction test; fewer, fuller facts. "
              "Runs on 8 GB machines."),
    _e(id="qwen2.5:7b", role="llm", provider="ollama", label="Qwen 2.5 7B",
       size_gb=4.7, min_ram_gb=16, damaged=7, tested=102, recommended=True,
       advice="Fast and faithful; occasionally drops a number. Best with 16 GB."),
    _e(id="qwen3:8b", role="llm", provider="ollama", label="Qwen 3 8B",
       size_gb=5.2, min_ram_gb=16, damaged=0, tested=15,
       advice="Faithful in a 15-sentence sample, but several times slower per memory "
              "than Gemma 3 4B."),
    _e(id="llama3.2", role="llm", provider="ollama", label="Llama 3.2 3B",
       size_gb=2.0, min_ram_gb=4, damaged=22, tested=102,
       advice="Small, but in SLM's test it often invented dates and numbers that were "
              "never said. Use only when nothing larger fits."),
)

LOCAL_EMBEDDERS: tuple[ModelEntry, ...] = (
    _e(id="nomic-ai/nomic-embed-text-v1.5", role="embedder",
       provider="sentence-transformers", label="Nomic Embed v1.5 (built in)",
       size_gb=0.55, min_ram_gb=4, dimension=768, recommended=True,
       advice="SLM's default. Runs inside SLM, no Ollama needed."),
    _e(id="nomic-embed-text", role="embedder", provider="ollama",
       label="Nomic Embed (Ollama)", size_gb=0.27, min_ram_gb=4, dimension=768,
       advice="The same model served by Ollama; same vectors, no re-index."),
)

#: Models the managed environment serves. Known to ``find`` (so a store built on one is
#: recognised) but not listed in ``catalog()``: they are offered by their own flow.
MANAGED_EMBEDDERS: tuple[ModelEntry, ...] = (
    _e(id="google/embeddinggemma-2", role="embedder", provider="slm-media",
       label="EmbeddingGemma 2 (managed)", size_gb=1.5, min_ram_gb=8, dimension=768,
       advice="One model for text and pictures, run by SLM's media environment. "
              "Switching to it re-indexes every memory."),
)

HOSTED_LLMS: tuple[ModelEntry, ...] = (
    _e(id="openai/gpt-6-luna", role="llm", provider="openrouter", label="GPT-6 Luna",
       price="$0.10 / $0.50", recommended=True, damaged=0, tested=102,
       advice="Lowest cost per memory; no changed facts in SLM's extraction test."),
    _e(id="deepseek/deepseek-v4.1-flash", role="llm", provider="openrouter",
       label="DeepSeek V4.1 Flash", price="$0.04 / $1.20",
       advice="Very low input cost, but in SLM's test it often added the conversation "
              "date and side facts nobody stated."),
    _e(id="anthropic/claude-haiku-4.5", role="llm", provider="openrouter",
       label="Claude Haiku 4.5", price="$1 / $5",
       advice="Reliable structured output at a moderate price."),
    _e(id="anthropic/claude-sonnet-5.5", role="llm", provider="openrouter",
       label="Claude Sonnet 5.5", price="$2 / $10",
       advice="Highest extraction quality; about twenty times the cost of the lowest."),
)

HOSTED_EMBEDDERS: tuple[ModelEntry, ...] = (
    _e(id="openai/text-embedding-3-small", role="embedder", provider="openrouter",
       label="OpenAI embedding 3 small", dimension=1536, price="$0.02 / -",
       advice="Hosted, inexpensive. Switching to it re-indexes every memory."),
    _e(id="qwen/qwen3-embedding-8b", role="embedder", provider="openrouter",
       label="Qwen3 Embedding 8B", price="$0.01 / -",
       advice="Hosted, long inputs. Its size is measured when chosen."),
)


def _base(model_id: str) -> str:
    """An Ollama tag without the default ``:latest`` suffix."""
    name = (model_id or "").strip()
    return name[:-len(":latest")] if name.endswith(":latest") else name


def find(model_id: str) -> ModelEntry | None:
    wanted = _base(model_id)
    for entry in (*LOCAL_LLMS, *LOCAL_EMBEDDERS, *MANAGED_EMBEDDERS, *HOSTED_LLMS, *HOSTED_EMBEDDERS):
        if entry.id == wanted:
            return entry
    return None


def _rank(entry: ModelEntry) -> tuple[float, str]:
    """Lower is better. (damaged + 1) / (tested + 2): a small sample with no
    damage does not outrank a large one with almost none."""
    if entry.damaged is None or not entry.tested:
        return (1.0, entry.id)  # untested sorts after every tested model
    return ((entry.damaged + 1) / (entry.tested + 2), entry.id)


def recommend_local_llms(ram_gb: float | None, installed: list[str]) -> list[Recommendation]:
    """Ollama language models for this computer, best first.

    Installed models that fit come first, ranked by SLM's extraction test;
    then installed models SLM has not tested; then catalogue models worth
    pulling that fit; installed models too large for this memory go last.
    Deterministic for the same inputs.
    """
    have = {_base(m) for m in installed if (m or "").strip()}
    have_llm = {m for m in have if not _is_embedder_name(m)}

    def fits(entry: ModelEntry | None) -> bool:
        return (entry is None or entry.min_ram_gb is None or ram_gb is None
                or ram_gb >= entry.min_ram_gb)

    known = {e.id: e for e in LOCAL_LLMS}
    out: list[Recommendation] = []
    for entry in sorted((known[m] for m in have_llm if m in known), key=_rank):
        if fits(entry):
            out.append(Recommendation(entry, entry.id, True, True, _tested_reason(entry)))
    for model_id in sorted(m for m in have_llm if m not in known):
        out.append(Recommendation(None, model_id, True, True,
                                  "Installed; SLM has not tested its extraction."))
    for entry in sorted(LOCAL_LLMS, key=_rank):
        if entry.id not in have_llm and entry.recommended and fits(entry):
            out.append(Recommendation(entry, entry.id, False, True,
                                      f"Not installed: ollama pull {entry.id}"))
    for entry in sorted((known[m] for m in have_llm if m in known), key=_rank):
        if not fits(entry):
            out.append(Recommendation(entry, entry.id, True, False,
                                      f"Needs about {entry.min_ram_gb} GB of memory."))
    return out


def best_local_llm(ram_gb: float | None, installed: list[str]) -> str:
    """The model the wizard pre-selects: the top installed, fitting choice, or
    the catalogue default when nothing suitable is installed."""
    for rec in recommend_local_llms(ram_gb, installed):
        if rec.installed and rec.fits:
            return rec.model_id
    return DEFAULT_LOCAL_LLM


def _tested_reason(entry: ModelEntry) -> str:
    return (f"{entry.damaged} of {entry.tested} test sentences gave a changed fact. "
            f"{entry.advice}")


def _is_embedder_name(model_id: str) -> bool:
    return "embed" in model_id.lower()


#: The Mode B template default when the installed models are not known yet.
#: The wizard and the dashboard replace it with ``best_local_llm`` of what is
#: actually installed before saving.
DEFAULT_LOCAL_LLM = "gemma3:4b"
#: The Mode C (OpenRouter) default: the lowest-cost tier (Varun, 2026-10-07).
DEFAULT_HOSTED_LLM = "openai/gpt-6-luna"
#: Direct-provider presets (no OpenRouter): the same lowest-cost choice per provider.
OPENAI_LLM = "gpt-6-luna"
ANTHROPIC_LLM = "claude-haiku-4-5"
OPENAI_EMBEDDING = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536


def as_dict(entry: ModelEntry) -> dict:
    return {f: getattr(entry, f) for f in entry.__slots__}


def catalog() -> dict:
    """Everything, for ``GET /api/v3/models/catalog`` and ``slm models``."""
    return {
        "version": CATALOG_VERSION,
        "local_llms": [as_dict(e) for e in LOCAL_LLMS],
        "local_embedders": [as_dict(e) for e in LOCAL_EMBEDDERS],
        "hosted_llms": [as_dict(e) for e in HOSTED_LLMS],
        "hosted_embedders": [as_dict(e) for e in HOSTED_EMBEDDERS],
        "defaults": {"local_llm": DEFAULT_LOCAL_LLM, "hosted_llm": DEFAULT_HOSTED_LLM},
    }


__all__ = [
    "CATALOG_VERSION", "DEFAULT_HOSTED_LLM", "DEFAULT_LOCAL_LLM", "HOSTED_EMBEDDERS",
    "HOSTED_LLMS", "LOCAL_EMBEDDERS", "LOCAL_LLMS", "MANAGED_EMBEDDERS", "ModelEntry", "Recommendation",
    "best_local_llm", "catalog", "find", "recommend_local_llms",
]
