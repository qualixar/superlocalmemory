# Pictures and documents: requirements and limits

Pictures and documents are off until you turn them on, in the dashboard (Settings, Images & documents) or with `slm media enable`. Turning them on downloads about 1.5 GB of models and tools into SLM's own folder. It does not change your existing memories or their embeddings: pictures and PDF pages are added next to them.

The **Upgrade memory engine (preview)** card is a separate, optional step. It re-reads every memory with the model that also reads pictures, so text and pictures share one search space. See [Upgrading the memory engine](#upgrading-the-memory-engine-preview) below.

## What your computer needs

| | |
|---|---|
| Memory (RAM) | 16 GB. Computers sold as 16 GB report a little less, so the check passes from 15 GiB. |
| Disk | About 1.5 GB for the models, plus what you save (up to 2 GB per profile). |
| Systems | macOS on Apple Silicon, Windows x64, Linux x86_64 and Linux ARM64 (aarch64) with a glibc such as Debian, Ubuntu or Fedora. Python 3.12, 3.13 or 3.14. |
| Not supported | Windows on ARM and Intel Macs. Text memory works there as before. |

Below 16 GB the switch is refused with the reason, because the picture model and a large PDF together can push a smaller machine into swapping. If you understand that risk, set `SLM_MEDIA_ALLOW_LOW_RAM=1` in the environment SLM starts from and turn the feature on again; SLM logs that warning once per run of the daemon (or terminal command), not on every save, so look for it near the start of the log.

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

The card in Settings shows the number of memories, a rough time estimate (not measured on your computer), and the memory and disk the upgrade needs. While it runs, recall keeps using the current engine. When it finishes, both switch in one step.

The previous engine's data stays on disk with no time limit, so **Roll back** is one click, until you choose **Free the old data**. Freeing it reclaims the disk space and ends the option to roll back.

## Housekeeping

`slm media gc` reports picture records whose memory is gone, picture files with no record, and pictures not yet linked to their memory (a save that was queued while the writer was busy). `slm media gc --apply` fixes them. It never deletes a memory, and it only ever touches stored picture files, never SLM's own databases.
