# Pictures and documents: requirements and limits

Pictures and documents are on by default. On a computer with 16 GB of memory or more, SLM sets them up by itself a few minutes after it starts, when the computer is idle (the Documents & Images pane and `slm media status` show the progress); on a smaller computer, turn them on in the dashboard or with `slm media enable`. Setting up downloads about 1.5 GB of models and tools into SLM's own folder. It does not change your existing memories or their embeddings: pictures and PDF pages are added next to them.

The **Upgrade memory engine (preview)** card is a separate, optional step, and not available for text yet (see below). It re-reads every memory with the model that also reads pictures, so text and pictures share one search space. See [Upgrading the memory engine](#upgrading-the-memory-engine-preview) below.

Three optional parts add to this: a short description of each picture (a 0.5 GB download), a larger model for scanned pages (1.9 GB) and "where text was found" (no download). See [Better pictures and documents](media-pack.md).

## What your computer needs

| | |
|---|---|
| Memory (RAM) | 16 GB or more for the automatic set-up (computers sold as 16 GB report a little less, so it starts from 15 GiB); below that it is one click, with a warning. |
| Disk | About 1.5 GB for the models, plus what you save (up to 2 GB per profile). |
| Systems | macOS on Apple Silicon, Windows x64, Linux x86_64 and Linux ARM64 (aarch64) with a glibc such as Debian, Ubuntu or Fedora. Python 3.12, 3.13 or 3.14. |
| Not supported | Windows on ARM and Intel Macs. Text memory works there as before. |

Below 16 GB nothing sets itself up and nothing is refused: turn pictures and documents on in the dashboard or with `slm media enable`, which prints "This computer has 8.0 GB of memory; pictures and documents may be slow and the models are released when memory runs low." (the picture model and a large PDF together can push a smaller machine into swapping, so SLM releases the models when memory runs low). To make a smaller computer take the automatic path too, set `SLM_MEDIA_ALLOW_LOW_RAM=1` in the environment SLM starts from; SLM logs a warning once per run of the daemon (or terminal command), not on every save, so look for it near the start of the log.

## Limits

| What | Limit |
|---|---|
| Picture formats | PNG, JPEG, GIF (first frame) and WebP |
| One picture | 25 MB from a file, the dashboard or an upload link; 8 MB as pasted data in a tool call from an app on this computer, and 512 KB when a remote app pastes it |
| One PDF | 100 MB (change with `SLM_DOC_MAX_MB`); 25 MB as pasted data in a tool call from an app on this computer, and 512 KB when a remote app pastes it |
| PDF pages | 500. A longer PDF is refused as a whole, not cut short. |
| Reading a PDF | 30 seconds per page and 20 minutes per document; the reader is stopped past 1.6 GB of memory |
| Picture and text reader memory | Checked while a request runs, twice a second: past 4.5 GB the reader is stopped, that request fails with a plain message, and the next one starts a fresh reader. On Linux the reader also starts with a hard limit above that, so one huge request fails inside the reader instead of taking the machine's memory. Text over 8,000 characters, a picture file over 25 MB or one with more than 50 million pixels is refused before it is sent. |
| Library | 2 GB of pictures and PDFs per profile |

A PDF that cannot be finished (password-protected, too many pages, a page that takes too long) is marked as not finished with the reason in plain words. Pages read before it stopped stay in your memory; removing the document hides them too. Dropping the same file again retries it.

## Saving from web apps

A web app connected through Web access can save a picture or PDF in two ways.

**Attached files (ChatGPT).** ChatGPT can hand an attached file straight to SLM. Your computer downloads it only from ChatGPT's own file hosts plus any host you add to `SLM_MEDIA_URL_HOSTS`; see [hosts](remote-access/hosts.md).

**One-time upload links (any web app).** The app asks SLM for a link and gives it to you; you open it and choose the file. A link:

- belongs to the app that asked for it, and is for one file. If the upload breaks partway, it can be started again up to 3 times; once a file is saved the link is spent;
- must be started within 10 minutes, and the upload must finish within 20 minutes of starting;
- is limited to 3 open links per connected app and 20 uploads per app per day;
- accepts only the kind of file it was made for (picture or PDF), checked from the file's first bytes on your computer.

Treat a link like a password for one upload: anyone who has it before it is used can save one file into your memory. If the picture tools are still starting when the upload finishes, the page says so; send the file again a minute later.

## Folders

A connected folder can include pictures and PDFs. Files found while pictures and documents were off are listed as skipped, and are read automatically on the next scan once the feature is ready.

## Upgrading the memory engine (preview)

**Not available for text yet.** Recall decides "this is in your memory" with an answer check calibrated per text model. Measured on the release harness, on questions it was not fitted on, the best check for EmbeddingGemma 2 found every real answer but also answered questions that have no answer (it refused 69% of made-up questions; the current engine refuses 98.5%), so the card and the command say why and do not start. `slm embedder switch google/embeddinggemma-2 --provider slm-media --dimension 768 --force` moves text anyway. If an earlier version already moved your text, the card and `slm doctor` say so; **Roll back** (or `slm embedder rollback`) goes back in minutes, because the previous vectors are kept. Pictures and documents use EmbeddingGemma 2 either way.

The card in Settings shows the number of memories, a rough time estimate (not measured on your computer), and the memory and disk the upgrade needs. While it runs, recall keeps using the current engine. When it finishes, both switch in one step.

The previous engine's data stays on disk with no time limit, so **Roll back** is one click, until you choose **Free the old data**. Freeing it reclaims the disk space and ends the option to roll back.

## Housekeeping

`slm media gc` reports picture records whose memory is gone, picture files with no record, and pictures not yet linked to their memory (a save that was queued while the writer was busy). `slm media gc --apply` fixes them. It never deletes a memory, and it only ever touches stored picture files, never SLM's own databases.
