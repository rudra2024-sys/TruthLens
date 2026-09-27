# Does provenance survive re-encoding?

Each image starts with an explicit AI declaration in its own metadata. Every degradation below is applied the way an editor or sharing pipeline would (decode, change pixels, re-encode); the file is then re-read.

## ChatGPT (C2PA Content Credentials) — 8 images

- Byte-for-byte copy (control): **8/8** still declare AI.
- Plain decode and re-save with no pixel change: **0/8**.

| degradation | still declares AI |
|---|---|
| jpeg 90 | 0/8 |
| jpeg 70 | 0/8 |
| jpeg 50 | 0/8 |
| jpeg 30 | 0/8 |
| jpeg 10 | 0/8 |
| resize 0.75 | 0/8 |
| resize 0.5 | 0/8 |
| resize 0.25 | 0/8 |
| blur 0.5 | 0/8 |
| blur 1 | 0/8 |
| blur 2 | 0/8 |
| blur 3 | 0/8 |
| noise 5 | 0/8 |
| noise 10 | 0/8 |
| noise 20 | 0/8 |
| crop 0.8 | 0/8 |
| crop 0.6 | 0/8 |
| social screenshot | 0/8 |
| social whatsapp | 0/8 |
| social webp | 0/8 |

## Portrait-app / DiffusionDB (prompt + seed text chunks) — 15 images

- Byte-for-byte copy (control): **4/15** still declare AI.
- Plain decode and re-save with no pixel change: **0/15**.

| degradation | still declares AI |
|---|---|
| jpeg 90 | 0/15 |
| jpeg 70 | 0/15 |
| jpeg 50 | 0/15 |
| jpeg 30 | 0/15 |
| jpeg 10 | 0/15 |
| resize 0.75 | 0/15 |
| resize 0.5 | 0/15 |
| resize 0.25 | 0/15 |
| blur 0.5 | 0/15 |
| blur 1 | 0/15 |
| blur 2 | 0/15 |
| blur 3 | 0/15 |
| noise 5 | 0/15 |
| noise 10 | 0/15 |
| noise 20 | 0/15 |
| crop 0.8 | 0/15 |
| crop 0.6 | 0/15 |
| social screenshot | 0/15 |
| social whatsapp | 0/15 |
| social webp | 0/15 |

**Reading:** the declaration lives in file metadata, not in the pixels, so anything that decodes and re-saves the image (a screenshot, a messaging app, a resize, a re-compression, an editor's "Save as") discards it. Provenance therefore only helps on files that reached the detector untouched, and its absence proves nothing. Caveat: some platforms deliberately preserve Content Credentials; this measures ordinary re-encoding.
