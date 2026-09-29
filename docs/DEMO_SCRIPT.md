# TruthLens — demo script (about 8 minutes)

A storyboard for presenting the project: what to click, what to say, and what you should see. Every expected result below was produced
during development on these exact files; if a number differs slightly on your machine (different CPU, a re-encoded file), say what you see —
don't claim the script's number.

**Story in one sentence:** *TruthLens gives a verdict, then shows its work — why, what the file's own metadata says, and where it can be
wrong — and it tells you honestly when it isn't sure.*

## 0. Setup (before you present)
1. Start the app (`docker compose up`, or the backend + `npm run dev` as in `CLAUDE.md` §6) and sign up with a demo account.
2. Prepare three helper files (from the repo root, one-time):
   ```bash
   # a "screenshot": the ChatGPT image re-saved, which drops its embedded credentials
   python -c "from PIL import Image; Image.open(r'backend/test_data/blind_spot_images/chatgpt_everyday/image1.png').convert('RGB').save('demo_resaved.png')"
   # a tampered copy: one byte flipped after the credentials were signed
   python -c "b=bytearray(open(r'backend/test_data/blind_spot_images/chatgpt_everyday/image1.png','rb').read()); b[len(b)//2]^=255; open('demo_tampered.png','wb').write(b)"
   ```
3. Have ready: an ordinary phone photo of your own, and one video (a short real one and a longer one for the progress bar).

## 1. The basics — a real photo (1 min)
**Upload your own photo.** Expect **AUTHENTIC / REAL**, high confidence.
Open **Provenance & Metadata**: *"No provenance metadata found — this says nothing either way."*
> *Say:* "Notice it doesn't claim this proves the photo is real — messaging apps strip metadata, so missing metadata means nothing."

## 2. A fake it catches — and shows why (1.5 min)
**Upload** `backend/test_data/blind_spot_images/portrait_app/part-000445__6188c86f-3cbd-455a-991c-c9da84b31da4.png` (the cat portrait).
Expect **MANIPULATED / FAKE, ~100 %**. Open **Why this verdict?**: the Grad-CAM heatmap lights up around the eyes and ears.
Open **Provenance**: *generation parameters embedded* (prompt and seed).
> *Say:* "The heatmap shows what the model's score depends on. It is an explanation of the model, not proof of where the edit is — the page says so."

## 3. The honest failure — and how the evidence layer helps (2 min) ★
**Upload** `backend/test_data/blind_spot_images/chatgpt_everyday/image1.png` (an AI-generated jewellery photo).
Expect the models to say **AUTHENTIC / REAL** (this is the documented blind spot: 0 of 8 such images caught).
**Directly under the verdict, a highlighted banner appears:** *"Provenance conflicts with the detection verdict."* Expand **Provenance**:
Content Credentials declare the file AI-generated (OpenAI, `gpt-image`), the signature is valid, and the file is unchanged since signing —
but the signer is not on the C2PA trust list, so its identity is self-asserted.
> *Say:* "Our detector was fooled at 100 % confidence — but the file carries its own declaration. We show both, and we say plainly that an
> untrusted signer isn't the same as a forged one."

**Now break it, honestly:** upload `demo_resaved.png` (the same image, re-saved). The credentials are gone; the models still say REAL.
> *Say:* "Provenance can be stripped by any re-save, so it can add evidence but never clear a file. We measured it: a byte-for-byte copy keeps the
> credentials (8/8); every re-encode we tried loses them (0/8)."

Upload `demo_tampered.png`: *"File was modified after its credentials were issued."*

## 4. Video with a real progress bar (1.5 min)
Upload a **longer** video. A progress bar shows real stages (*Reading frames (k/16)*, *Finding faces*, *Scoring frames*) — while it runs,
open another tab and note the site stays responsive. Click **Cancel analysis** on one scan to show it stops and stores nothing.
Open **Why this verdict?** on a finished video: the bar chart of per-frame scores, the exact face crops the model saw, and heatmaps on the
most suspicious frames.
> *Say:* "For one consumer-app fake (Akool) the model scored every frame near 50 %; the crops show the face detector locked onto a wall poster instead of the
> face. That's a concrete, visible reason for the miss." (Only if you use that file: `fake video 3.mp4` → UNCERTAIN 48 %.)

## 5. Close the loop (30 s)
Click **No, it is wrong** on a result, pick what it really is, leave "allow keeping this file" **unticked** (or tick it), send. Then **Withdraw feedback**.
> *Say:* "Feedback is opt-in for keeping the file, changeable, and withdrawable — and only consenting files can ever be exported for evaluation."

## 6. The report (30 s)
**Download PDF Report** — verdict, per-model scores, the heatmap or per-frame chart, the provenance section (with the conflict box from §3), and the caveats.

## 7. The numbers slide (1 min) — say these first-hand, not oversold
| | Result |
|---|---|
| Image, on data like its training data | **98.4 %** accuracy, AUC 1.00 |
| Image, on generators it **never saw** | **53 %** accuracy, AUC 0.67, only ~27 % of fakes caught |
| Video, unseen face swaps (RTFS) | 79 % accuracy, AUC 0.89 |
| Provenance | catches 9 of the 1 026 AI images the models miss — all 8 ChatGPT ones — but only if the file is untouched |
| Robustness | Screenshots, WhatsApp-style and WebP: AUC stays ≥ 0.99. But faint noise or heavy JPEG makes ~30–40 % of *real* photos look fake (the CLIP sub-model is the cause); heavy blur hides fakes (the ConvNeXt sub-model). |
> *Say:* "It is excellent on what it was trained on and weak on what it hasn't seen. We measured that instead of hiding it — the next step is
> training data with more generators."

## Questions you should expect
| Question | Honest answer |
|---|---|
| Why not just 100 % accurate? | Detectors generalise poorly to new generators; ours drops from 98 % to 53 % on unseen ones. The UNCERTAIN band, the explanations and the provenance layer exist because of that. |
| Why `max` of two models instead of the average? | It recovers the portrait-app genre CLIP learned, at ~1 point cost on ordinary sources (measured; `CLAUDE.md` §3). |
| Can I trust "REAL"? | No. It means "no evidence found", not "proven authentic" — 0 of 8 ChatGPT images were caught. |
| Can I trust "FAKE"? | It is a lead; about 14 % of real photos on unseen sources were wrongly flagged. |
| What about audio? | Pretrained AASIST is integrated; a teammate is training a separate audio model. |
| Where does the data go? | Files stay on the server you run; keeping them for improvement is opt-in per result. There is currently no auto-delete (`THREAT_MODEL.md`). |
| What would you do next? | Train with more generators (including consumer face-swap apps), add data-retention controls, and fuse provenance into the verdict (needs a deliberate decision). |
