# TruthLens — architecture

How the system is put together, how a scan flows through it, and where each guarantee is enforced. Diagrams are
[Mermaid](https://mermaid.js.org/) (rendered by GitHub). For the authoritative list of what is real ML vs heuristic vs dead code,
see [`CLAUDE.md`](../CLAUDE.md); numbers on how well it works are in [`ETHICS_AND_LIMITATIONS.md`](ETHICS_AND_LIMITATIONS.md).

## 1. System overview

```mermaid
flowchart LR
    U["User's browser<br/>React + Vite single-page app"] -->|"HTTPS"| N["nginx<br/>static files and /api proxy"]
    N -->|"/api/v1/*"| A["FastAPI backend<br/>one process"]

    subgraph BE["Backend process"]
        A --> R["Routers<br/>auth, upload, detect, jobs,<br/>report, dashboard, feedback"]
        R --> D["Detectors<br/>image, video, audio"]
        R --> X["Explain and provenance<br/>services"]
        R --> J["Job manager<br/>background video scans"]
        J --> D
        D --> M["Models in memory<br/>ConvNeXt-Tiny + CLIP head<br/>EfficientNet-B0<br/>AASIST via ONNX"]
        X --> M
        R --> P["PDF report generator<br/>ReportLab"]
    end

    R --> DB[("SQLite database")]
    R --> FS[("Uploaded files<br/>and reports on disk")]
    X --> FS
    D --> FS
```

* **One backend process on purpose.** Background-job progress lives in that process's memory (results are durable in the
  database). See [DEPLOYMENT.md](DEPLOYMENT.md).
* **Models are loaded once** per process (first request pays the load) and shared by the detectors and the explainers.
* **Everything is per-user.** Every route that touches an upload, result, job, explanation, provenance or feedback checks
  ownership and answers **404 (not 403)** for someone else's data, so an id never reveals that it exists.

## 2. The three detectors

```mermaid
flowchart TB
    subgraph IMG["Image  (services/image/detector.py)"]
        I0["Image file"] --> I1["ConvNeXt-Tiny<br/>P(fake)"]
        I0 --> I2["CLIP ViT-B/16 frozen + trained head<br/>P(fake)"]
        I1 --> I3{"max of the two"}
        I2 --> I3
        I3 --> I4["Verdict band around 0.5<br/>REAL / UNCERTAIN / FAKE"]
    end
    subgraph VID["Video  (services/video, Video Model v1)"]
        V0["Video file"] --> V1["16 evenly spaced frames"]
        V1 --> V2["Haar face crop + 20% padding<br/>224 px, ImageNet normalisation"]
        V2 --> V3["EfficientNet-B0 per frame"]
        V3 --> V4["Mean of frame logits, sigmoid"]
        V4 --> V5["Verdict band around 0.525"]
    end
    subgraph AUD["Audio  (services/audio, pretrained AASIST)"]
        A0["Audio file"] --> A1["AASIST ONNX<br/>windowed spoof score"]
        A1 --> A2["Verdict band around 0.5"]
    end
```

* Image uses **max**, not average: it lets either model's "fake" through, which recovers the portrait-app genre CLIP learned, at
  a small cost in false alarms on ordinary photos (measured, see the evaluation summary).
* Video Model v1 is verified against the original evaluation script's frame selection, face crop and pooling; a regression
  script guards that parity. The old byte-entropy heuristic remains selectable (`VIDEO_MODEL_BACKEND=heuristic`).
* The **UNCERTAIN band** (±0.2 around the threshold) exists so a borderline score is never presented as a confident call.

## 3. A scan, end to end

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant A as API
    participant D as Detector
    participant DB as Database
    B->>A: POST /upload (file)
    A->>DB: store upload row + file on disk
    B->>A: POST /detect/{id}  (image / audio)
    A->>D: run detector
    D->>DB: result row + per-model scores
    A-->>B: verdict, confidence, sub-scores
    par optional, on demand
        B->>A: GET /detect/{id}/explain
        A-->>B: Grad-CAM heatmap or per-frame scores
    and
        B->>A: GET /detect/{id}/provenance
        A-->>B: C2PA credentials, metadata, ELA
    and
        B->>A: PUT /detect/{id}/feedback
        A->>DB: store "was this correct?"
    and
        B->>A: GET /report/{id}
        A-->>B: PDF forensic report
    end
```

`POST /detect/{id}` is **idempotent**: an upload has at most one result. (A repeated call once stored a second row and broke every
read endpoint for that upload; there is a regression test.)

### Video scans run as background jobs

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant A as API
    participant J as Job manager
    participant W as Worker thread
    B->>A: POST /detect/{id}/jobs
    A->>J: submit (idempotent per upload)
    A-->>B: 202 job (queued)
    J->>W: run video detector in a worker thread
    loop until finished
        W-->>J: progress (frame k of 16, stage)
        B->>A: GET /jobs/{job}
        A-->>B: state, progress, stage, queue position
    end
    W->>A: result stored in the database
    B->>A: GET /detect/{id}/result
    Note over B,J: DELETE /jobs/{job} cancels: an exception raised inside the progress callback stops the scan, nothing is stored
```

* At most `DETECTION_JOB_CONCURRENCY` jobs run at once (default 1: inference is CPU-bound); others wait with a queue position.
* The scoring runs in a worker thread, so a long video **no longer freezes the server** for other users.
* Progress numbers are real (frames decoded / cropped), not a timer. The callbacks are observation-only: predictions are
  bit-identical with and without them (tested).

## 4. Evidence beyond the verdict

| Layer | What it adds | Never does |
|---|---|---|
| **Explainability** (`services/explain`) | Grad-CAM heatmap (image) or per-frame scores + the exact face crops (video) | Change a verdict; explain CLIP (no heatmap for it, and the UI says so) |
| **Provenance** (`services/provenance`) | C2PA Content Credentials (signature, unchanged-since-signing, trust), EXIF, embedded generation parameters, ELA visual aid | Treat missing metadata as evidence; alter a stored result |
| **Feedback** (`api/routes/feedback.py`) | Users say whether a verdict was right; opt-in to let the file be kept for evaluation | Keep a file without explicit consent |

Where provenance and the model disagree (models say REAL, the file declares itself AI-generated) all three surfaces — API, UI and
PDF — say so explicitly.

## 5. Data model

```mermaid
erDiagram
    USER ||--o{ UPLOAD : owns
    UPLOAD ||--o| DETECTION_RESULT : "has at most one"
    DETECTION_RESULT ||--o| IMAGE_ANALYSIS : "image scores"
    DETECTION_RESULT ||--o| VIDEO_ANALYSIS : "video scores"
    DETECTION_RESULT ||--o| AUDIO_ANALYSIS : "audio scores"
    UPLOAD ||--o{ FEEDBACK : receives
    USER ||--o{ FEEDBACK : gives

    USER {
        string user_id PK
        string email UK
        string password_hash
        bool is_active
    }
    UPLOAD {
        string upload_id PK
        string user_id FK
        string media_type
        string storage_url
        float file_size_kb
    }
    DETECTION_RESULT {
        string result_id PK
        string upload_id FK
        string verdict
        float confidence_score
        string model_used
    }
    FEEDBACK {
        string feedback_id PK
        string upload_id FK
        string user_id FK
        bool agrees
        string true_label
        bool allow_reuse
    }
```

Feedback is unique per (upload, user). Background jobs are **not** in the database (in-memory, single process).

## 6. Code map

| Area | Where |
|---|---|
| API routes | `backend/app/api/routes/` (`auth`, `upload`, `detect`, `jobs`, `report`, `dashboard`, `feedback`) |
| Detectors | `backend/app/services/{image,video,audio}/detector.py`; video model in `services/video/model_v1/` |
| Model code | `backend/app/pipelines/{image,image_clip}/` |
| Explain / provenance | `backend/app/services/explain/`, `backend/app/services/provenance/` |
| Jobs | `backend/app/services/jobs.py` |
| PDF report | `backend/app/services/report/generator.py` |
| Evaluation harness | `backend/eval/` (metrics, robustness, provenance study, results) |
| Tests / CI | `backend/tests/`, `.github/workflows/ci.yml` |
| Frontend | `frontend/src/` (`App.jsx` routes, `components/UploadFlow.jsx` the scan flow) |
| Deployment | `docker-compose.yml` (dev), `docker-compose.prod.yml`, `backend/Dockerfile.prod`, `frontend/nginx.prod.conf` |
