"""Build the comparison dataset under $SLM_BENCH_HOME/dataset.

LoCoMo text is read from the downloaded file at build time; only ids are kept in
golden/locomo_selection.json (the data is CC BY-NC 4.0 and never committed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SEED = 1729
DEFAULT_QUOTAS = {"text_single_hop": 20, "text_multi_hop": 14, "entity": 14,
                  "temporal": 18, "unanswerable": 12}
CATEGORY = {"multi_hop": 1, "temporal": 2, "single_hop": 4, "adversarial": 5}
ENTITY_STARTS = ("who", "which", "where", "what")
DIA = re.compile(r"D\d+:\d+")


def bench_home() -> Path:
    return Path(os.environ.get("SLM_BENCH_HOME", Path.home() / ".cache" / "slm-embedder-compare"))


def dia_ids(evidence: list[str]) -> list[str]:
    """Normalise evidence strings such as 'D8:6; D9:17' to dia ids."""
    return [m for e in evidence for m in DIA.findall(e)]


def turns_of(sample: dict) -> dict[str, dict]:
    """dia_id -> {speaker, text, date} for every turn of a conversation."""
    conv, out = sample["conversation"], {}
    for key, turns in conv.items():
        if re.fullmatch(r"session_\d+", key) and isinstance(turns, list):
            date = _short_date(conv.get(f"{key}_date_time", ""))
            for t in turns:
                out[t["dia_id"]] = {"speaker": t["speaker"], "text": t["text"], "date": date}
    return out


def _short_date(raw: str) -> str:
    m = re.search(r"(\d{1,2} [A-Za-z]+),? (\d{4})", raw)
    return f"{m.group(1)} {m.group(2)}" if m else raw.strip()


def _usable(sample: dict, qa: dict, turns: dict) -> bool:
    ev = dia_ids(qa.get("evidence", []))
    return bool(ev) and all(e in turns for e in ev)


def _is_entity(qa: dict) -> bool:
    words = qa["question"].split()
    return words[0].lower() in ENTITY_STARTS and any(w[:1].isupper() for w in words[1:])


def select_locomo(data: list[dict], quotas: dict[str, int], n_convs: int = 3,
                  seed: int = SEED) -> list[dict]:
    """Seeded id-only selection from the conversations richest in needed categories."""
    usable = {s["sample_id"]: {i for i, qa in enumerate(s["qa"])
                               if _usable(s, qa, turns_of(s)) and qa["category"] in (1, 2, 4)
                               or qa["category"] == 5}
              for s in data}
    ranked = sorted(usable, key=lambda sid: (-len(usable[sid]), sid))[:n_convs]
    pools = {name: [] for name in quotas}
    for s in data:
        if s["sample_id"] not in ranked:
            continue
        for i in sorted(usable[s["sample_id"]]):
            qa = s["qa"][i]
            cat = qa["category"]
            key = (s["sample_id"], i)
            if cat == 5:
                pools["unanswerable"].append(key)
            elif cat == 2:
                pools["temporal"].append(key)
            elif cat == 1:
                pools["text_multi_hop"].append(key)
            elif cat == 4:
                pools["text_single_hop"].append(key)
    rng, chosen, out = random.Random(seed), set(), []
    entity_pool = [k for n in ("text_single_hop", "text_multi_hop") for k in pools[n]]
    sample_of = {s["sample_id"]: s for s in data}
    for name in ("text_single_hop", "text_multi_hop", "temporal", "unanswerable", "entity"):
        pool = entity_pool if name == "entity" else pools[name]
        pool = [k for k in pool if k not in chosen]
        if name == "entity":
            pool = [k for k in pool if _is_entity(sample_of[k[0]]["qa"][k[1]])]
        for k in sorted(rng.sample(pool, min(quotas[name], len(pool)))):
            chosen.add(k)
            out.append({"sample_id": k[0], "qa_index": k[1], "stratum": name})
    return sorted(out, key=lambda r: (r["stratum"], r["sample_id"], r["qa_index"]))


def _locomo_corpus(samples: dict[str, dict]) -> list[dict]:
    corpus = []
    for sid, sample in sorted(samples.items()):
        for dia, t in turns_of(sample).items():
            corpus.append({"doc_id": f"t:{sid}:{dia}", "kind": "text",
                           "text": f"[{t['date']}] {t['speaker']}: {t['text']}",
                           "meta": {"sample_id": sid, "dia_id": dia}})
    return corpus


def _locomo_queries(selection: list[dict], samples: dict[str, dict], doc_ids: set[str]):
    queries, qrels = [], {}
    for row in selection:
        sid, idx = row["sample_id"], row["qa_index"]
        qa = samples[sid]["qa"][idx]
        qid = f"loc:{sid}:{idx}"
        answerable = row["stratum"] != "unanswerable"
        rel = {}
        if answerable:
            for dia in dia_ids(qa.get("evidence", [])):
                doc = f"t:{sid}:{dia}"
                if doc not in doc_ids:
                    raise ValueError(f"{qid}: evidence turn {dia} is missing from the corpus")
                rel[doc] = 1
            if not rel:
                raise ValueError(f"{qid}: answerable query without evidence")
        queries.append({"id": qid, "text": qa["question"], "stratum": row["stratum"],
                        "answerable": answerable, "provisional": True})
        qrels[qid] = rel
    return queries, qrels


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


ARXIV_IDS = ("2603.14588", "2603.02601", "2604.04514")


def arxiv_pdfs(arxiv_dir: Path | None) -> list[Path]:
    """The author's public arXiv PDFs, only when every one of them was downloaded."""
    if arxiv_dir is None:
        return []
    paths = [Path(arxiv_dir) / f"{i}.pdf" for i in ARXIV_IDS]
    return paths if all(p.is_file() and p.stat().st_size > 0 for p in paths) else []


def _pdf_pages(pdf_path: Path, pdf_name: str, ds: Path, corpus: list, page_text: dict) -> None:
    import ocr
    import pypdfium2 as pdfium
    import synth_pdfs

    n_pages = len(pdfium.PdfDocument(str(pdf_path)))
    for p in range(1, n_pages + 1):
        doc_id = f"pdf:{pdf_name}#p{p}"
        png = f"media/pdf/{pdf_name}_p{p}.png"
        ocr.render_page(pdf_path, p - 1, ds / png)
        corpus.append({"doc_id": doc_id, "kind": "pdf_page", "path": png,
                       "meta": {"pdf": f"media/pdf/{pdf_name}.pdf", "page": p}})
        if pdf_name == synth_pdfs.IMAGE_ONLY_NAME:
            page_text[doc_id] = _norm(" ".join(synth_pdfs.IMAGE_ONLY_PAGES[p - 1]))
        else:
            page_text[doc_id] = _norm(ocr.pdf_text_layer(pdf_path, p - 1))


def coco_files(coco_dir: Path | None) -> tuple[Path, Path] | None:
    """(captions csv, images zip) when both were downloaded."""
    if coco_dir is None:
        return None
    csv_path, zip_path = Path(coco_dir) / "test_5k.csv", Path(coco_dir) / "images.zip"
    return (csv_path, zip_path) if csv_path.is_file() and zip_path.is_file() else None


def _coco(ds: Path, files: tuple[Path, Path], selection: list[dict]) -> tuple[list, list, dict]:
    """Real photos; a query is the photo's first human caption, verbatim (stratum 'photo')."""
    import ast
    import csv
    import zipfile

    rows = {int(r["cocoid"]): r for r in csv.DictReader(files[0].open(encoding="utf-8"))}
    out = ds / "media" / "coco"
    out.mkdir(parents=True, exist_ok=True)
    corpus, queries, qrels = [], [], {}
    with zipfile.ZipFile(files[1]) as zf:
        names = {Path(n).name: n for n in zf.namelist() if not n.startswith("__MACOSX")}
        for sel in selection:
            doc_id = f"img:coco_{sel['cocoid']}"
            (out / sel["filename"]).write_bytes(zf.read(names[sel["filename"]]))
            corpus.append({"doc_id": doc_id, "kind": "image", "path": f"media/coco/{sel['filename']}",
                           "meta": {"source": "coco2014-test"}})
            if sel["query"]:
                qid = f"coco:{sel['cocoid']}"
                caption = ast.literal_eval(rows[sel["cocoid"]]["raw"])[sel["caption_index"]]
                queries.append({"id": qid, "text": " ".join(caption.split()), "stratum": "photo",
                                "answerable": True, "provisional": True})
                qrels[qid] = {doc_id: 1}
    return corpus, queries, qrels


def _media_corpus(ds: Path, arxiv: list[Path] = ()) -> tuple[list[dict], dict[str, str]]:
    """Copy screenshots, generate synthetic images and PDFs, render PDF pages."""
    import synth_images
    import synth_pdfs

    media, corpus, page_text = ds / "media", [], {}
    media.mkdir(parents=True, exist_ok=True)
    for src in sorted((REPO_ROOT / "docs" / "screenshots").rglob("*.png")):
        rel = src.relative_to(REPO_ROOT / "docs" / "screenshots").with_suffix("")
        name = "shot-" + "-".join(rel.parts)
        shutil.copyfile(src, media / f"{name}.png")
        corpus.append({"doc_id": f"img:{name}", "kind": "image", "path": f"media/{name}.png",
                       "meta": {"source": "docs/screenshots"}})
    for name in synth_images.generate(media / "syn"):
        corpus.append({"doc_id": f"img:syn_{name}", "kind": "image", "path": f"media/syn/{name}.png",
                       "meta": {"source": "synthetic"}})
    for pdf_name in synth_pdfs.generate(media / "pdf", REPO_ROOT):
        _pdf_pages(media / "pdf" / f"{pdf_name}.pdf", pdf_name, ds, corpus, page_text)
    for src in arxiv:
        name = f"arxiv-{src.stem}"
        shutil.copyfile(src, media / "pdf" / f"{name}.pdf")
        _pdf_pages(media / "pdf" / f"{name}.pdf", name, ds, corpus, page_text)
    return corpus, page_text


def _media_queries(path: Path, doc_ids: set[str], page_text: dict[str, str]):
    queries, qrels = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        anchor = row.pop("anchor", None)
        if row["answerable"]:
            missing = [d for d in row["relevant"] if d not in doc_ids]
            if missing:
                raise ValueError(f"{row['id']}: unknown doc ids {missing}")
        if anchor:
            hits = [d for d, t in page_text.items() if _norm(anchor) in t]
            if hits != row["relevant"]:
                raise ValueError(f"{row['id']}: anchor found on {hits}, expected {row['relevant']}")
        qrels[row["id"]] = {d: 1 for d in row.pop("relevant")}
        queries.append(row)
    return queries, qrels


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def manifest_hash(test_jsonl: Path, test_qrels: Path) -> str:
    h = hashlib.sha256()
    h.update(test_jsonl.read_bytes())
    h.update(b"\n--\n")
    h.update(test_qrels.read_bytes())
    return h.hexdigest()


def dataset_hash(ds: Path) -> str:
    """Hash of everything a run depends on: corpus, both query splits, both qrels."""
    h = hashlib.sha256()
    for rel in ("corpus.jsonl", "queries/dev.jsonl", "queries/test.jsonl",
                "qrels/dev.json", "qrels/test.json"):
        h.update((Path(ds) / rel).read_bytes())
        h.update(b"\n--\n")
    return h.hexdigest()


def check_frozen(digest: str, frozen_path: Path | None, refreeze: bool) -> None:
    """Refuse to change the committed test-manifest hash unless refreeze is set."""
    if frozen_path is None:
        return
    frozen_path = Path(frozen_path)
    current = frozen_path.read_text().strip() if frozen_path.exists() else None
    if current == digest:
        return
    if not refreeze:
        raise ValueError(
            f"test queries/qrels hash {digest[:12]} differs from the frozen "
            f"{(current or 'missing')[:12]} in {frozen_path}; rerun with --refreeze to accept")
    frozen_path.write_text(digest + "\n")


def build(locomo_path: Path, selection_path: Path, out_dir: Path, media: bool = True,
          media_queries_path: Path | None = None, frozen_path: Path | None = None,
          refreeze: bool = False, arxiv_dir: Path | None = None,
          coco_dir: Path | None = None) -> dict:
    """Build into a scratch dir, check the frozen test hash, then replace out_dir."""
    out_dir = Path(out_dir)
    scratch = out_dir.with_name(out_dir.name + ".building")
    if scratch.exists():
        shutil.rmtree(scratch)
    summary = _build_into(locomo_path, selection_path, scratch, media, media_queries_path,
                          arxiv_pdfs(arxiv_dir), coco_files(coco_dir))
    digest = manifest_hash(scratch / "queries" / "test.jsonl", scratch / "qrels" / "test.json")
    try:
        check_frozen(digest, frozen_path, refreeze)
    except ValueError:
        shutil.rmtree(scratch)
        raise
    if out_dir.exists():
        shutil.rmtree(out_dir)
    scratch.rename(out_dir)
    return summary


def _build_into(locomo_path: Path, selection_path: Path, out_dir: Path, media: bool,
                media_queries_path: Path | None, arxiv: list[Path] = (),
                coco: tuple[Path, Path] | None = None) -> dict:
    """Write corpus, split queries/qrels and manifest.lock; return per-stratum counts."""
    data = json.loads(Path(locomo_path).read_text())
    selection = json.loads(Path(selection_path).read_text())
    samples = {s["sample_id"]: s for s in data if s["sample_id"] in {r["sample_id"] for r in selection}}
    out_dir = Path(out_dir)
    (out_dir / "queries").mkdir(parents=True)
    (out_dir / "qrels").mkdir()
    corpus = _locomo_corpus(samples)
    page_text: dict[str, str] = {}
    if media:
        extra, page_text = _media_corpus(out_dir, arxiv)
        corpus += extra
    doc_ids = {d["doc_id"] for d in corpus}
    queries, qrels = _locomo_queries(selection, samples, doc_ids)
    if media:
        mq, mqr = _media_queries(media_queries_path or HERE / "golden" / "media_queries.jsonl",
                                 doc_ids, page_text)
        queries, qrels = queries + mq, {**qrels, **mqr}
        if arxiv:
            aq, aqr = _media_queries(HERE / "golden" / "arxiv_queries.jsonl", doc_ids, page_text)
            queries, qrels = queries + aq, {**qrels, **aqr}
        if coco:
            selection_c = json.loads((HERE / "golden" / "coco_selection.json").read_text())
            cc, cq, cqr = _coco(out_dir, coco, selection_c)
            corpus, queries, qrels = corpus + cc, queries + cq, {**qrels, **cqr}
    dev, test = stats.stratified_split(queries)
    _dump_jsonl(out_dir / "corpus.jsonl", corpus)
    for name, part in (("dev", dev), ("test", test)):
        _dump_jsonl(out_dir / "queries" / f"{name}.jsonl", sorted(part, key=lambda q: q["id"]))
        sub = {q["id"]: qrels[q["id"]] for q in sorted(part, key=lambda q: q["id"])}
        (out_dir / "qrels" / f"{name}.json").write_text(json.dumps(sub, sort_keys=True, indent=1))
    digest = manifest_hash(out_dir / "queries" / "test.jsonl", out_dir / "qrels" / "test.json")
    (out_dir / "manifest.lock").write_text(json.dumps({"sha256": digest, "n_test": len(test)}) + "\n")
    counts: dict[str, int] = {}
    for q in queries:
        counts[q["stratum"]] = counts.get(q["stratum"], 0) + 1
    return {"strata": counts, "total": len(queries), "dev": len(dev), "test": len(test),
            "corpus": len(corpus)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--selection", type=Path, default=HERE / "golden" / "locomo_selection.json")
    ap.add_argument("--write-selection", action="store_true",
                    help="(re)generate the id-only selection from the downloaded LoCoMo file")
    ap.add_argument("--no-media", action="store_true")
    ap.add_argument("--refreeze", action="store_true",
                    help="accept a changed test set and rewrite golden/test_manifest.sha256")
    args = ap.parse_args(argv)
    locomo = args.home / "locomo" / "data" / "locomo10.json"
    if args.write_selection:
        sel = select_locomo(json.loads(locomo.read_text()), DEFAULT_QUOTAS)
        args.selection.write_text(json.dumps(sel, indent=0) + "\n")
    try:
        summary = build(locomo, args.selection, args.home / "dataset", media=not args.no_media,
                        frozen_path=HERE / "golden" / "test_manifest.sha256", refreeze=args.refreeze,
                        arxiv_dir=args.home / "arxiv", coco_dir=args.home / "coco")
    except ValueError as exc:
        print(f"refusing to build: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
