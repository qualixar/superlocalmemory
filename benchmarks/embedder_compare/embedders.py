"""Embedders. torch / sentence-transformers are imported lazily inside load()."""
from __future__ import annotations

import os
from typing import Protocol

import numpy as np

NOMIC_ID = "nomic-ai/nomic-embed-text-v1.5"
EG2_ID = "google/embeddinggemma-2"


class Embedder(Protocol):
    name: str

    def load(self) -> None: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def embed_docs(self, texts: list[str]) -> np.ndarray: ...
    def embed_images(self, paths: list[str]) -> np.ndarray: ...


def _unit(vectors) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class Nomic:
    """nomic-embed-text-v1.5 exactly as SLM ships it: raw text, no task prefixes."""

    name = "nomic"
    query_prefix = ""
    doc_prefix = ""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(NOMIC_ID, trust_remote_code=True, device="cpu")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([self.query_prefix + t for t in texts]))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode([self.doc_prefix + t for t in texts]))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError("nomic is text only; images go through OCR text")


class NomicPrefixed(Nomic):
    """Reference only: nomic with its recommended task prefixes (not what SLM ships)."""

    name = "nomic_prefixed"
    query_prefix = "search_query: "
    doc_prefix = "search_document: "


class EG2Full:
    """EmbeddingGemma 2, full model (text and images, one 768-d space).

    CPU settings: sdpa attention and float32. An earlier measurement on this
    class of machine saw about 46 s per image idle with default kwargs and
    1,094 ms with these settings; this harness does not re-measure that.
    The audio tower is never loaded.
    """

    name = "eg2_full"
    config_kwargs: dict | None = {"audio_config": None}

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        kwargs = {"model_kwargs": {"attn_implementation": "sdpa", "dtype": torch.float32}}
        if self.config_kwargs:
            kwargs["config_kwargs"] = self.config_kwargs
        self._model = SentenceTransformer(EG2_ID, device="cpu", **kwargs)

    def _prompt(self, name: str) -> str:
        prompts = getattr(self._model, "prompts", None) or {}
        if name not in prompts:
            raise KeyError(f"model has no prompt named {name!r} (has {sorted(prompts)})")
        return name

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts, prompt_name=self._prompt("SearchQuery")))

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts, prompt_name=self._prompt("Document")))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        return _unit([self._model.encode({"image": p}) for p in paths])


class EG2Text(EG2Full):
    """EmbeddingGemma 2 text-only loadout (vision and audio towers not loaded)."""

    name = "eg2_text"
    config_kwargs = {"vision_config": None, "audio_config": None}

    def embed_images(self, paths: list[str]) -> np.ndarray:
        raise NotImplementedError("text-only loadout cannot embed images")


# Under SLM's pins (transformers 5.10.4) the hub config fails strict validation
# (n_inner is stored as 2048.0); point this at a local copy whose config has n_inner=2048.
NOMIC_VISION_ID = os.environ.get("SLM_BENCH_NOMIC_VISION", "nomic-ai/nomic-embed-vision-v1.5")
QWEN_VL_ID = "Qwen/Qwen3-VL-Embedding-2B"


def _layer_norm(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    v = v - v.mean(axis=1, keepdims=True)
    return v / np.sqrt(v.var(axis=1, keepdims=True) + 1e-5)


def _load_nomic_vision(path: str):
    """Build the vision tower on CPU and load its weights strictly.

    Under transformers 5.10.4, from_pretrained of the remote NomicVisionModel fails
    (no post_init, so no all_tied_weights_keys); calling post_init then loads but
    yields NaN vectors (buffers built on the meta device). Plain construction plus
    a strict state-dict load matches the repo's own ONNX export (cosine 1.0).
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    local = path if os.path.isdir(path) else snapshot_download(path)
    cfg = AutoConfig.from_pretrained(local, trust_remote_code=True)
    model = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], local)(cfg)
    model.load_state_dict(load_file(os.path.join(local, "model.safetensors")), strict=True)
    return model.eval()


class NomicMultimodal(Nomic):
    """Candidate C1: nomic-embed-text-v1.5 as SLM ships it plus nomic-embed-vision-v1.5.

    Text queries and docs stay exactly as shipped (raw text), so an existing store
    needs no re-index. Media-channel queries follow the vision card: "search_query: "
    prefix, mean pooling, layer norm, unit length. Images: CLS token, unit length.
    """

    name = "nomic_c1"

    def load(self) -> None:
        from transformers import AutoImageProcessor

        super().load()
        self._processor = AutoImageProcessor.from_pretrained(NOMIC_VISION_ID)
        self._vision = _load_nomic_vision(NOMIC_VISION_ID)

    def embed_media_queries(self, texts: list[str]) -> np.ndarray:
        raw = self._model.encode(["search_query: " + t for t in texts], normalize_embeddings=False)
        return _unit(_layer_norm(raw))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        with torch.no_grad():
            out = self._vision(**self._processor(images, return_tensors="pt")).last_hidden_state
        return _unit(out[:, 0].numpy())


class QwenVL:
    """Candidate C3: Qwen3-VL-Embedding-2B (Apache 2.0), one 2048-d space for text and images.

    CPU, float32, sdpa attention. Queries carry a retrieval instruction (the card
    recommends task instructions); docs and images use the model's default one.
    Images are capped at 262,144 px (about 256 visual tokens, close to EmbeddingGemma 2's
    default 280); the model's own default allows 1,310,720 px (about 1,280 tokens), which
    costs roughly five times the CPU time per image.
    """

    name = "qwen3vl"
    query_prompt = "Retrieve images or text relevant to the user's query."
    max_pixels = 262_144

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            QWEN_VL_ID, device="cpu",
            model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32},
            processor_kwargs={"max_pixels": self.max_pixels})

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts, prompt=self.query_prompt))

    embed_media_queries = embed_queries

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return _unit(self._model.encode(texts))

    def embed_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        return _unit(self._model.encode([Image.open(p).convert("RGB") for p in paths]))


REGISTRY = {"nomic": Nomic, "nomic_prefixed": NomicPrefixed, "eg2_text": EG2Text, "eg2_full": EG2Full,
            "nomic_c1": NomicMultimodal, "qwen3vl": QwenVL}
