# Better pictures and documents (the optional models)

Three parts on top of [pictures and documents](pictures-and-documents.md), all **on by default**. On a computer with 16 GB of memory or more, the running SLM background service sets up pictures and documents and then downloads both models by itself, one at a time, a few minutes after it has started and settled. On a computer with less than 16 GB, nothing downloads by itself: the dashboard (**Documents & Images**, the card **Better pictures and documents**) and `slm media pack enable` give a one-click download instead, with a plain warning that pictures and documents may be slow there (see below). Two of the parts download a model; its size is shown before anything is downloaded by hand. Nothing is downloaded by a recall, a hook, a one-off command or a test. A part you turned off stays off: the automatic download never turns on anything you switched off. The models use the libraries the pictures-and-documents install already has, so there is no new runtime to install.

| Part | What it does | Download | Model |
|---|---|---|---|
| Describe pictures (`captions`, flag `media.captions`) | Writes a short description of each picture you save, on macOS, Windows and Linux. macOS keeps its own text reading and subject labels too. | 0.5 GB | `florence-community/Florence-2-base`, MIT, pinned to one revision |
| Read scanned documents better (`documents`, flag `media.document_pack`) | Reads scanned PDF pages with a larger model. The standard reader (PP-OCR) stays the default and is the fallback for any page the larger model cannot read. | 1.9 GB | `PaddlePaddle/PaddleOCR-VL-1.6`, Apache-2.0, pinned to one revision |
| Remember where text was found (`regions`, flag `media.regions`) | Keeps the page and the place each piece of read text came from. | none | none |

The sizes are the true sizes of the pinned revisions on the Hugging Face hub on 18 November 2026. They are shown to you, not promised: a hub change cannot alter a pinned revision.

```bash
slm media pack status                    # each part, its state and its download size
slm media pack enable captions           # shows "Describe pictures: a 0.5 GB download", asks; --yes skips the question
slm media pack enable documents --yes
slm media pack enable regions
slm media pack disable captions          # off again; memories are kept (--remove-files also deletes the model)
```

Free disk space of twice the download is needed; without it the automatic download waits and says so. The kill switch (`SLM_4230_OFF=1`, or the dashboard) turns all three off at once; with a part off, saving a picture or a document behaves exactly as it did before.

## What a description is

With `captions` on, saving a picture adds one short line of words about what it shows, under the label `[Description written by a model]`. It is searched like any text of the picture, and recall shows it with that label, so you can tell a model's words from your own and from the text read out of the picture. Every memory saved from a picture has the origin `model_generated` (the bottom of the origin order, for models' words), and the description joins it under the same origin. It goes through the same redaction as text read from a picture. The receipt of a save, the `remember_media` tool and the upload routes return it as `description` (only when there is one).

A description never blocks a save. If the model is not there, is still starting, is too large for the memory free right now, or fails, the picture is saved without a description and the kind of problem is written to the log (never the picture or its words). A person on this computer waits for a cold start of the model (about a minute the first time); a web app's request does not.

## Where text was found

With `regions` on, reading a picture or a scanned page also keeps the box of each line it read (as fractions of the width and height, so a thumbnail and the original agree) and a redacted snippet of its words, in `media.db` next to the picture. `slm why <fact_id>`, `fetch(why=true)`, `GET /api/trust/why/{fact_id}` and the dashboard's **Why was this shown?** then say where the fact's words sat: "from page 3, top left", "from bottom of the picture". A fact is matched to the lines it quotes (at least 60% of a line's words appear in the fact); a fact made from a description or from your own note matches none and shows none. The boxes come from the standard reader (RapidOCR). Apple Vision text carries no boxes, so on a Mac the places appear for pictures read by RapidOCR only. The document model reports one region for a page, the page itself: its layout boxes need a separate layout model that this part does not load, so for a page it read you see "from page 3".

Erasing a picture or a document, or deleting a profile, erases its regions (a database trigger and the profile tables of `media.db`); a profile fold moves them with the profile. There is no new migration: the table is part of `media.db`, which creates it when it opens.

## 16 GB and more, or less

| This computer | What happens |
|---|---|
| 16 GB or more (15 GiB, so machines sold as 16 GB count) | Automatic: the background service downloads the picture model, then each part, one at a time, when the computer is idle and has the disk space. If a download fails it tries again later, with longer and longer waits, and `slm media pack status` says why in plain words. |
| Less than 16 GB, or an unknown amount | One click: nothing downloads by itself. `slm media enable`, `slm media pack enable` and the dashboard buttons work, and show: "This computer has 8.0 GB of memory; pictures and documents may be slow and the models are released when memory runs low." |
| You turned a part off (`slm media disable`, `slm media pack disable`, the dashboard switch, `SLM_FEATURE_MEDIA_CAPTIONS=off`) or the kill switch is on | Nothing downloads, ever, until you turn it on again. |

The warning is honest: both pinned models were loaded in their native `transformers` classes and produced a picture description and a page reading on a CPU (19 November 2026); the test run peaked at about 6 GB of memory. On a small computer that leaves little room, so SLM gives the memory back when the machine runs short. `SLM_NO_AUTO_DOWNLOAD=1` stops the automatic download (the test suite sets it); `SLM_MEDIA_ALLOW_LOW_RAM=1` is the developer switch that lets a small computer take the automatic path.

## Memory and time

Each model runs alone in its own worker process: only one model is in memory in that process, it never shares memory with the picture-search model, and the worker is stopped a minute after its last use, so its memory goes back to the machine. Loading waits for the same memory reservation the picture model uses, and it is refused rather than started when the machine is short of memory. The caps are 2.5 GB for Florence-2-base and 6.5 GB for PaddleOCR-VL; a measured CPU run of both peaked at about 6 GB.

## Known limits

- Both pinned models were loaded in their native `transformers` classes and produced a picture description and a page reading on a CPU (19 November 2026), with a peak of about 6 GB of memory. `tests/test_media/test_pack_models_native.py` repeats that when `SLM_TEST_PACK_CAPTIONS_DIR` and `SLM_TEST_PACK_DOCUMENTS_DIR` name folders with the pinned files; the unit and scenario tests use a fake model at the worker boundary.
- Speed on a CPU is not yet published as a figure.
- Models are loaded from a local folder only (`local_files_only`), and `trust_remote_code` is never set, so the Python files some model repositories carry are never run.
