"""Text of the TruthLens final project report (VIT B.Tech blackbook).

build_report.py renders this into the VIT template. Markup inside strings: **bold**, *italic*, citations {key} or
{key1,key2} (numbered IEEE-style by first appearance). Every number comes from backend/eval/results/, video_retrain/,
image_retrain/ or CLAUDE.md; where a figure is a smoke test or a small sample, the text says so.

Block types:
  ("h1", title) ("h2", title) ("h3", title) ("p", text) ("bul", [items]) ("num", [items])
  ("fig", key, caption, width_in)            key -> figures/<key>.png or screens/<crop key>
  ("tab", caption, header, rows, widths_dxa)  widths sum to <= 8666
  ("eq", omml_spec, label)                    see build_report.eq()
"""

TITLE = "TruthLens: An Evidence-Oriented Multimodal Platform for Detecting AI-Generated Media with Proctored Identity Verification"
SHORT_TITLE = "TruthLens: An Evidence-Oriented Multimodal Platform for Detecting AI-Generated Media"

# (Surname Firstname, roll number placeholder). Roll numbers are not in the repository: fill them in.
STUDENTS = [("Jaiswal Manini", "Roll No. ______"), ("Shinde Rudrapratap", "Roll No. ______"),
            ("Pethkar Atharva", "Roll No. ______"), ("Thakur Soham", "Roll No. ______")]
GUIDE = "Prof. Dhanashree Tamhane"
YEAR = "2025-2026"

ABSTRACT = (
    "Image generators, face-swap apps and voice-cloning services have made synthetic media cheap to produce and hard to "
    "recognise by eye. This project presents TruthLens, a working web platform that examines an uploaded image, video or "
    "audio file and returns a banded verdict (REAL, UNCERTAIN or FAKE) together with the evidence behind it. Images are "
    "scored by a fine-tuned ConvNeXt-Tiny classifier, a frozen CLIP ViT-B/16 probe and, when a face is present, a face-swap "
    "check, combined by a maximum rule; videos by an EfficientNet-B0 applied to sixteen face crops with Platt-calibrated "
    "scores and a resolution guard; speech by a fine-tuned wav2vec 2.0 model over 4-second windows. Every verdict comes "
    "with Grad-CAM maps or per-frame and per-window scores, C2PA Content Credential validation, metadata and Stable "
    "Diffusion watermark checks, calibrated confidence and a PDF report. A second module, ProctorLens, verifies a "
    "candidate against a government ID and monitors proctored online examinations with eight behavioural signals. The "
    "deployed models were evaluated separately on data sources seen and unseen in training: the image ensemble reaches "
    "98.4% accuracy on 4,000 seen-source images but 53.2% on 2,517 images from unseen generators; the round-4 video model "
    "raises held-out ROC-AUC from 0.765 to 0.961; identity matching reaches 96.5% accuracy on 1,000 LFW pairs. The results "
    "show that detector scores must be presented as one strand of evidence rather than a final answer."
)
KEYWORDS = ("Deepfake detection, AI-generated images, multimodal media forensics, explainable AI, content provenance, "
            "online proctoring")

ACK = [
    "We express our sincere gratitude to our project guide, **Prof. Dhanashree Tamhane**, for her constant guidance, "
    "patient reviews and the many suggestions that shaped this work from the first proposal to the final review.",
    "We thank the Head of the Department of Information Technology and the Principal of Vidyalankar Institute of "
    "Technology for providing the facilities and the encouragement needed to carry out a project of this scope. We are "
    "also grateful to the faculty members and laboratory staff of the department for their help during the reviews and "
    "for access to the laboratory resources.",
    "We acknowledge the authors and maintainers of the public datasets and open-source libraries this project builds on, "
    "including CIFAKE, DeepDetect-2025, RealTimeFaceSwap-10k, FaceForensics++, Celeb-DF, DiffusionDB, PyTorch, OpenCLIP, "
    "Hugging Face Transformers, facenet-pytorch, the C2PA SDK and FastAPI. Free GPU time on Kaggle and Google Colab made "
    "the fine-tuning rounds possible.",
    "Finally, we thank our families and friends for their support and patience throughout the project.",
]

ABBREVIATIONS = [
    ("AASIST", "Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention"),
    ("AI", "Artificial Intelligence"), ("API", "Application Programming Interface"),
    ("AUC", "Area Under the (ROC) Curve"), ("C2PA", "Coalition for Content Provenance and Authenticity"),
    ("CI", "Continuous Integration"), ("CLIP", "Contrastive Language-Image Pre-training"),
    ("CNN", "Convolutional Neural Network"), ("CSP", "Content-Security-Policy"), ("DFD", "Data Flow Diagram"),
    ("DWT", "Discrete Wavelet Transform"), ("ECE", "Expected Calibration Error"), ("ELA", "Error Level Analysis"),
    ("EXIF", "Exchangeable Image File Format"), ("FF++", "FaceForensics++"),
    ("Grad-CAM", "Gradient-weighted Class Activation Mapping"), ("GAN", "Generative Adversarial Network"),
    ("JWT", "JSON Web Token"), ("LFW", "Labeled Faces in the Wild"), ("MCQ", "Multiple-Choice Question"),
    ("MTCNN", "Multi-task Cascaded Convolutional Networks"), ("ONNX", "Open Neural Network Exchange"),
    ("PDF", "Portable Document Format"), ("RMS", "Root Mean Square"), ("ROC", "Receiver Operating Characteristic"),
    ("RTFS", "RealTimeFaceSwap-10k dataset"), ("SD", "Stable Diffusion"), ("SPA", "Single-Page Application"),
    ("VAD", "Voice Activity Detection"), ("ViT", "Vision Transformer"), ("XMP", "Extensible Metadata Platform"),
    ("YOLO", "You Only Look Once (object detector)"),
]

BODY = [
    # =========================================================================================== 1
    ("h1", "Introduction"),
    ("p", "Until a few years ago, producing a convincing fake photograph took skill and hours of editing. Today it takes a "
          "one-line text prompt. Diffusion models render photorealistic scenes on a laptop, consumer face-swap apps replace a "
          "face in a phone video within minutes, and voice-cloning services need only a short sample of the target speaker. "
          "Journalists, fact-checkers, teachers, recruiters and ordinary users are increasingly asked to decide whether a file "
          "is genuine, usually without any forensic training and without access to the tools researchers use."),
    ("p", "TruthLens is a web platform built to help with that decision. A user uploads an image, a video or an audio clip; "
          "the system routes it to a detector trained for that kind of media and returns a verdict of REAL, UNCERTAIN or FAKE "
          "with a confidence score. Unlike a bare classifier, every verdict is accompanied by the evidence behind it: heat-maps "
          "of the regions the model relied on, per-frame or per-window scores, the file's own provenance metadata, and a "
          "downloadable PDF forensic report. The same face-analysis stack is reused by **ProctorLens**, a module that verifies "
          "a candidate's identity against a government ID and monitors a proctored online examination."),
    ("p", "This report documents the complete system as submitted for the final review: its design, the three detection "
          "pipelines and how each one parses its input, the evidence layer, the proctoring module, the implementation and "
          "deployment, and a measured evaluation that deliberately reports where the system fails as well as where it works."),

    ("h2", "Background"),
    ("p", "Synthetic media is produced by three broad families of generative models. **Generative adversarial networks** "
          "(GANs) such as StyleGAN produce faces and objects by training a generator against a discriminator. **Diffusion "
          "models** such as Stable Diffusion {ldm}, Midjourney and DALL-E iteratively denoise random noise into an image "
          "conditioned on a text prompt, and now dominate image generation. **Face-swap and re-enactment systems**, from the "
          "autoencoder methods in FaceForensics++ {ffpp} to modern one-shot swappers such as inswapper, transplant one "
          "person's face onto another's head in every frame of a video. For audio, text-to-speech and voice-conversion "
          "systems clone a speaker's voice from seconds of reference audio {asvspoof2019}."),
    ("p", "Detection research has grown alongside, but it is split by modality, each with its own benchmarks: FaceForensics++ "
          "{ffpp} and Celeb-DF {celebdf} for face video, CIFAKE {cifake} and GenImage {genimage} for images, and ASVspoof "
          "{asvspoof2019} for speech. Most detectors are deep convolutional or transformer networks trained as binary "
          "real-versus-fake classifiers. A complementary line of work, **content provenance**, attaches cryptographically "
          "signed records of how a file was made: the C2PA standard {c2pa} lets cameras and generators sign a manifest that "
          "travels with the file, and OpenAI, Adobe and others now embed such credentials in their outputs."),

    ("h2", "Motivation"),
    ("p", "Three observations motivated the project. First, deepfakes have moved from novelty to everyday harm: fake "
          "evidence in disputes, non-consensual imagery, voice-clone fraud calls and impersonation in online interviews and "
          "examinations. Second, the free detectors available to the public usually return a single number with no "
          "explanation, so the user cannot judge whether to trust it, and they rarely say how they were evaluated. Third, "
          "published detectors are known to transfer poorly to generators they were not trained on {wang2020,ojha2023,genimage}, "
          "so headline benchmark accuracy overstates what a user will actually experience."),
    ("p", "We therefore set out to build a system that is useful to a non-expert *and* honest about its limits: one that "
          "covers all three media types, explains each decision, uses provenance where it exists, refuses to force a "
          "borderline call, and is evaluated in a way that keeps familiar data apart from unfamiliar data. Online examinations, "
          "where a candidate may be impersonated or replaced by a live deepfake, are a natural second application of the same "
          "face-analysis components and became the ProctorLens module."),

    ("h2", "Problem Statement"),
    ("p", "Given an image, video or audio file supplied by a user, determine whether it is likely to have been generated or "
          "manipulated by AI, and present that judgement together with inspectable evidence and a calibrated statement of "
          "confidence. The system must handle borderline cases explicitly, must not treat missing metadata as proof of "
          "authenticity, must keep every user's data private, and its accuracy must be measured separately on data similar to "
          "and different from its training data. In addition, for proctored online examinations, verify that the candidate "
          "matches a government-issued ID and continuously flag behaviour that suggests impersonation or malpractice."),

    ("h2", "Objectives"),
    ("p", "The objectives of this project are:"),
    ("bul", [
        "To study existing techniques for detecting AI-generated images, face-swap videos and synthetic speech, together "
        "with explanation, provenance and calibration methods.",
        "To design and implement a multimodal detection platform with a dedicated, trained or fine-tuned detector for each "
        "media type and a common verdict, report and history layer.",
        "To add an evidence layer: Grad-CAM and saliency explanations, C2PA and metadata provenance, invisible-watermark "
        "detection and calibrated confidence, without letting missing provenance count as authenticity.",
        "To build ProctorLens: government-ID verification, live identity matching and continuous monitoring of proctored "
        "multiple-choice examinations with server-side enforcement.",
        "To evaluate the deployed models separately on seen and unseen data sources, and to measure robustness to everyday "
        "degradation, explanation faithfulness, provenance survival, calibration and error rates across demographic groups.",
        "To harden the system for deployment: authentication, owner-only access, upload validation, rate limits, automated "
        "tests, CI and containerised production configuration.",
    ]),

    ("h2", "Scope of the Work"),
    ("p", "**In scope.** Still images (JPEG, PNG, WebP, HEIC/HEIF), videos (MP4, MOV, WebM and similar containers) and audio "
          "(WAV, MP3, M4A, OGG, Opus, WebM) up to 100 MB; binary classification into real or AI-generated/manipulated with a "
          "three-way banded verdict; per-modality explanations; provenance checks; PDF reports; per-user history, dashboard "
          "and feedback; identity verification and proctoring of MCQ examinations created by an organisation; Docker-based "
          "deployment on a single CPU server."),
    ("p", "**Out of scope and known limits.** The video model targets face-swap and lip-sync deepfakes; fully generated "
          "video (for example Sora or Kling) is outside what it was trained on. Detection generalises poorly to image "
          "generators absent from the training data. The system produces leads and evidence for human review, not proof, and "
          "is not intended for legal, employment or disciplinary decisions without a human in the loop. Inference runs on "
          "CPU in a single server process; horizontal scaling would need a shared job store."),

    ("h2", "Organization of the Report"),
    ("p", "The rest of the report is organized as follows. Chapter 2 reviews the related literature and identifies the "
          "research gaps. Chapter 3 presents the proposed methodology: the architecture, the datasets, and each pipeline "
          "(image, video, audio, evidence layer and ProctorLens) step by step. Chapter 4 describes the system design and "
          "implementation: requirements, UML and data-flow diagrams, database schema, API, frontend, security, testing and "
          "deployment. Chapter 5 discusses the experimental results with screenshots of the working system. Chapter 6 "
          "concludes the report and outlines future work, and Chapter 7 lists the publication arising from the project."),

    # =========================================================================================== 2
    ("h1", "Literature Survey"),
    ("p", "This chapter reviews recent work on the three detection problems and on the supporting techniques the project "
          "uses. For each work we note the method, the data, the reported result and the limitation that matters for a "
          "deployed system."),
    ("h2", "Review of Existing Work"),
    ("h3", "AI-Generated Image Detection"),
    ("p", "Wang et al. {wang2020} trained a ResNet-50 on images from a single GAN (ProGAN) with blur and JPEG augmentation "
          "and showed that it transfers surprisingly well to other GANs of that period, establishing augmentation as a key "
          "ingredient. Frank et al. {frank2020} showed that the up-sampling layers of GANs leave regular artefacts in the "
          "frequency spectrum that a simple classifier can exploit, although such cues are fragile to re-compression. With "
          "diffusion models the picture changed: the GenImage benchmark {genimage} (eight generators, over a million images) "
          "reports large accuracy drops when a detector is tested on generators it was not trained on. Ojha et al. {ojha2023} "
          "proposed classifying frozen CLIP {clip} features with a light head; because the feature space is not shaped around "
          "one generator, it generalises markedly better than end-to-end training. Bird and Lotfi {cifake} released CIFAKE, "
          "a balanced 120,000-image dataset of CIFAR-10 photographs and Stable Diffusion 1.4 images at 32×32 pixels, and "
          "reached about 93% accuracy with a CNN, but the images are far smaller than real uploads. ConvNeXt {convnext} "
          "modernised the ResNet design with transformer-style choices and is a strong, efficient backbone for fine-tuning."),
    ("h3", "Face-Manipulation Video Detection"),
    ("p", "Rössler et al. {ffpp} introduced FaceForensics++, with four manipulation methods (Deepfakes, Face2Face, FaceSwap, "
          "NeuralTextures) and an XceptionNet baseline on face crops that is highly accurate on high-quality in-distribution "
          "data but degrades under compression and on unseen methods. Celeb-DF {celebdf} provided higher-quality celebrity "
          "swaps on which earlier detectors' AUC dropped sharply. The DFDC dataset {dfdc} added scale and diversity, and its "
          "strongest entries were ensembles of EfficientNet {efficientnet} models applied to face crops. WildDeepfake "
          "{wilddeepfake} collected deepfakes from the internet and showed a further gap between laboratory and real-world "
          "performance. A recurring weakness of the face-crop design is the face detector itself: if it locks onto the wrong "
          "region, the classifier never sees the manipulation."),
    ("h3", "Synthetic Speech Detection"),
    ("p", "The ASVspoof 2019 challenge {asvspoof2019} established logical-access benchmarks of text-to-speech and voice-"
          "conversion attacks. Jung et al. {aasist} proposed AASIST, which models spectral and temporal sub-graphs with "
          "graph attention directly from raw waveforms and achieves very low equal-error rates on ASVspoof 2019 LA. Tak et al. "
          "{tak2022} showed that a self-supervised wav2vec 2.0 front-end {wav2vec2}, fine-tuned with data augmentation, "
          "improves generalisation considerably. Müller et al. {inthewild} collected the In-the-Wild dataset of real public "
          "recordings and found that models that excel on ASVspoof degrade sharply on real-world audio, which motivates "
          "training on diverse sources rather than one benchmark."),
    ("h3", "Explanation, Provenance and Calibration"),
    ("p", "Grad-CAM {gradcam} weights the last convolutional feature maps by the gradient of the target class to produce a "
          "coarse localisation of the evidence a CNN used; input-gradient saliency {simonyan2014} is a simpler alternative for "
          "models without a convolutional map. Neither proves manipulation; both show what the score depends on. Provenance "
          "answers a different question: the C2PA specification {c2pa} defines signed manifests that record a file's origin "
          "and edits, and Stable Diffusion's reference pipeline embeds a fixed invisible watermark {ldm,invisiblewm}. Error "
          "level analysis {krawetz2007} visualises JPEG re-compression differences but is easily misread. Guo et al. {guo2017} "
          "showed that modern networks are over-confident and that temperature scaling is a simple, effective fix; Platt "
          "scaling {platt1999} fits a logistic map from scores to probabilities."),
    ("h3", "Face Verification and Online Proctoring"),
    ("p", "MTCNN {mtcnn} jointly detects faces and five facial landmarks with a cascade of three small CNNs. FaceNet "
          "{facenet} maps a face to a compact embedding in which Euclidean or cosine distance measures identity, and "
          "InceptionResNet-v1 models trained on VGGFace2 {vggface2} are widely used for verification; LFW {lfw} remains the "
          "standard verification benchmark. Single-stage detectors of the YOLO family {yolov8} detect everyday objects such as "
          "phones and books in real time. Commercial proctoring tools combine such components but rarely publish thresholds or "
          "error rates, and automated flags are known to raise fairness and privacy concerns, which argues for transparent, "
          "reviewable signals rather than automatic verdicts."),

    ("h2", "Summary of Literature"),
    ("p", "Table 2.1 summarizes the literature reviewed."),
    ("tab", "Summary of literature survey",
     ["Sr. No.", "Author(s) & Year", "Method / Technique", "Dataset", "Result", "Limitation"],
     [["1", "Rössler et al., 2019 {ffpp}", "XceptionNet on face crops", "FaceForensics++", "High accuracy on HQ in-distribution fakes", "Drops under compression and unseen methods"],
      ["2", "Li et al., 2020 {celebdf}", "Benchmark of improved face swaps", "Celeb-DF v2", "Existing detectors' AUC falls sharply", "Dataset only; celebrity faces"],
      ["3", "Wang et al., 2020 {wang2020}", "ResNet-50 + blur/JPEG augmentation", "ProGAN, 11 GANs", "Transfers across GANs", "Weaker on diffusion models"],
      ["4", "Frank et al., 2020 {frank2020}", "Frequency (DCT) spectrum cues", "GAN images", "Up-sampling artefacts are detectable", "Fragile to re-compression"],
      ["5", "Jung et al., 2022 {aasist}", "AASIST graph attention on raw audio", "ASVspoof 2019 LA", "Very low EER on benchmark", "Degrades on real-world audio"],
      ["6", "Tak et al., 2022 {tak2022}", "wav2vec 2.0 front-end + augmentation", "ASVspoof 2021 LA/DF", "Better generalisation", "Heavy model, benchmark-bound"],
      ["7", "Müller et al., 2022 {inthewild}", "Evaluation of 12 audio detectors", "In-the-Wild", "Large drop on real recordings", "Exposes, does not solve, the gap"],
      ["8", "Ojha et al., 2023 {ojha2023}", "Frozen CLIP ViT + linear/NN head", "GAN + diffusion", "Better unseen-generator accuracy", "Still misses some generators"],
      ["9", "Zhu et al., 2023 {genimage}", "Million-image benchmark", "GenImage (8 generators)", "Cross-generator accuracy drops", "Format biases (JPEG vs PNG)"],
      ["10", "Bird & Lotfi, 2024 {cifake}", "CNN on 32×32 images", "CIFAKE", "About 93% accuracy", "Tiny images, one generator"]],
     [560, 1500, 1720, 1300, 1700, 1886]),

    ("h2", "Research Gaps"),
    ("p", "The review points to the following gaps, which this project addresses:"),
    ("bul", [
        "**Single-modality tools.** Published detectors cover one media type; a user facing an image, a video and a voice "
        "note needs three separate tools with different outputs.",
        "**Bare scores.** Most tools return a probability with no explanation, no statement of uncertainty and no "
        "indication of how reliable the number is.",
        "**Seen-versus-unseen reporting.** Accuracy is usually reported on held-out parts of the training benchmark, which "
        "measures fit rather than generalisation to new generators.",
        "**Provenance in isolation.** C2PA credentials and watermarks are rarely combined with detector scores, and the "
        "question of what to show when the two disagree is seldom addressed.",
        "**Robustness and calibration.** Behaviour under ordinary sharing (re-compression, resizing, messaging apps) and "
        "the calibration of confidence are rarely measured for a deployed system.",
        "**Opaque proctoring.** Proctoring products seldom publish their identity thresholds or flag logic, and do not check "
        "whether the face on camera is itself synthetic.",
    ]),

    # =========================================================================================== 3
    ("h1", "Proposed Methodology"),
    ("p", "TruthLens follows one principle throughout: a detector score is one strand of evidence, not the final answer. "
          "Each upload is routed to a modality-specific detector, the score is mapped to a three-way verdict, and an "
          "independent evidence layer adds explanations and provenance that the user can inspect. This chapter describes the "
          "architecture, the data, and each pipeline in the order in which it processes a file."),

    ("h2", "System Architecture"),
    ("p", "The block diagram of the proposed system is shown in Figure 3.1. A React single-page application talks to a "
          "FastAPI backend through an nginx reverse proxy. The backend is organised in layers: routers (authentication, "
          "upload, detection, background jobs, reports, dashboard, feedback, identity, exams and administration) validate "
          "requests and enforce ownership; detectors and ProctorLens services run the models; the evidence layer computes "
          "explanations, provenance and calibration on demand; and the storage layer keeps results in SQLite through async "
          "SQLAlchemy and files on disk. Models are loaded once per process and shared between detectors and explainers."),
    ("fig", "arch", "Block diagram of the proposed system", 6.0),
    ("p", "Two design decisions are visible in the figure. First, the evidence layer is separate from the detectors: an "
          "explanation or provenance failure can never change or break a stored verdict. Second, slow work (video scans and "
          "proctoring checks) runs in background job managers with a bounded thread pool, so a long video never freezes the "
          "server for other users."),

    ("h2", "Dataset Description"),
    ("p", "The details of the datasets used are given in Table 3.1 (training) and Table 3.2 (evaluation). A source is "
          "called *seen* when the model was trained on data from it, so accuracy on it measures fit; *unseen* sources were "
          "never used for training and measure generalisation. The datasets themselves are not stored in the repository; "
          "only small test samples are."),
    ("tab", "Dataset description: training sources",
     ["Parameter", "Image detector", "Video detector", "Audio detector"],
     [["Name of dataset", "CIFAKE; AI-vs-Human-Generated; 140k Real and Fake Faces; DeepDetect-2025; DiffusionDB portrait subset", "FaceForensics++, Celeb-DF, DFDC, DFD, RTFS-10k, FakeAVCeleb, phone-recorded swaps (round 4)", "ASVspoof 2019 (VC attacks), DEEP-VOICE, In-the-Wild, LibriSpeech, VCC 2020"],
      ["Source", "Kaggle; Hugging Face (DiffusionDB, CC0)", "Research releases, Kaggle, Hugging Face, team recordings", "Challenge releases, Kaggle, OpenSLR"],
      ["Number of samples", "120,000 + ~80,000 + 140,000 + 100,000+ images; 257 hand-reviewed portrait-app fakes (CLIP only)", "Up to 600 crops per class per dataset per epoch; 8–16 frames per video", "Clip-level, split by speaker or by conversion system"],
      ["Number of classes", "2 (real, AI-generated)", "2 (real, face-swap)", "2 (genuine, deepfake)"],
      ["Train / Test split", "Dataset test folders; exact hold-out split not recorded", "Grouped by source clip and person; 926-video validation split", "Speaker-split / system-split held-out sets"]],
     [1500, 2400, 2400, 2366]),
    ("tab", "Dataset description: evaluation sources",
     ["Source", "Files", "Status", "Used for"],
     [["CIFAKE, AI-vs-Human, 140k Faces, DeepDetect (test folders)", "4,000", "Seen", "Image fit"],
      ["GenImage subset: real / SD / Midjourney / BigGAN", "600 / 300 / 300 / 300", "Unseen", "Image generalisation"],
      ["OpenFake (2025 generators): real / fake", "500 / 500", "Unseen", "Image generalisation"],
      ["ChatGPT \"everyday\" photos; team photos; portrait-app images", "8; 9; 15", "Unseen / CLIP only", "Blind spots"],
      ["RTFS-10k: real / inswapper / uniface", "150 / 75 / 75", "Unseen (epoch 11)", "Video"],
      ["356 local held-out videos (round 4 comparison)", "185 real / 171 fake", "Held out", "Video round 4"],
      ["UTKFace real portraits, 10 gender × race groups", "600", "Unseen", "Fairness"],
      ["LFW DevTest pairs (same / different person)", "500 / 500", "Unseen", "Identity threshold"],
      ["RWFS real-app face swaps + real photos", "500 / 500", "Unseen (r1)", "Face-swap check"]],
     [3700, 1900, 1500, 1566]),
    ("p", "Several dataset quirks affect how results must be read: a trivial JPEG statistic separates DeepDetect's real "
          "and fake images with AUC 0.995 (a compression-history shortcut); in GenImage the real images are JPEG and the "
          "fakes PNG; CIFAKE is natively 32×32; and RTFS real and fake clips went through different encodes."),

    ("h2", "Algorithm / Model Description"),
    ("p", "The workflow common to all media types is:"),
    ("num", [
        "Step 1: The user uploads a file; the server checks its size and its leading bytes (magic-byte sniffing) against the "
        "declared media type, stores it and creates an upload record owned by that user.",
        "Step 2: Detection is dispatched by media type. Images and audio are scored within the request; videos run as a "
        "background job that reports real progress.",
        "Step 3: The detector returns a fake probability p, which is mapped to a REAL / UNCERTAIN / FAKE verdict, and a "
        "calibrated confidence is computed.",
        "Step 4: On demand, the evidence layer computes the explanation and the provenance signals; if provenance declares "
        "AI generation while the models do not, a conflict note is attached.",
        "Step 5: The verdict, sub-scores, evidence and caveats are shown in the interface and rendered into a PDF report; "
        "the user may give feedback.",
    ]),

    ("h3", "Image Pipeline"),
    ("p", "Figure 3.2 shows how an image is parsed. After decoding with Pillow (HEIC/HEIF through pillow-heif), three "
          "scorers run in parallel on the same file:"),
    ("fig", "pipe_image", "Image detection pipeline", 6.2),
    ("bul", [
        "**ConvNeXt-Tiny** {convnext}, fine-tuned by the team. It was first trained on CIFAKE, then fine-tuned on four "
        "sources (CIFAKE, AI-vs-Human, 140k Faces and DeepDetect-2025, which covers StyleGAN3, DALL-E 3, Midjourney and SD3) "
        "with random JPEG re-compression, blur and resize augmentation. The image is resized directly to 224×224 with bicubic "
        "interpolation and ImageNet normalisation. An earlier variant first shrank every image to 32×32 to match CIFAKE; on a "
        "held-out evaluation of 3,000 images per source it cut face-image accuracy from 99.8% to 57%, so it was removed.",
        "**CLIP ViT-B/16** {clip}, with the backbone frozen and only a small MLP head trained (512 → 256 → 2, dropout 0.3), "
        "following {ojha2023}. Round 3 of the head added 257 hand-reviewed Stable Diffusion portraits retrieved from "
        "DiffusionDB {diffusiondb} by prompt text, after both sub-models were found to call stylised \"portrait-app\" images "
        "REAL with over 99.9% confidence. CLIP's accuracy on that genre rose from 26.7% to 93.3%.",
        "**Face-swap check.** Whole-image models are near chance on face swaps, because a swap is a real photograph with only "
        "one face replaced. If a frontal face at least 64 px wide is found (Haar cascade first, MTCNN as fallback), the face "
        "crop is scored by an EfficientNet-B0 photo model (round 2: a 50/50 weight blend of the video model and a fine-tune "
        "on real-app swaps) and Platt-calibrated. Any failure here means only \"check not applied\"; the scan still succeeds.",
    ]),
    ("p", "With p_C, p_L and p_F the fake probabilities of ConvNeXt, CLIP and the face check, the image score is given by "
          "Equation (3.1):"),
    ("eq", [("sub", "p", "img"), ("t", "=max("), ("sub", "p", "C"), ("t", ","), ("sub", "p", "L"), ("t", ","), ("sub", "p", "F"), ("t", ")")], "3.1"),
    ("p", "The maximum rather than the mean was chosen deliberately. ConvNeXt was not retrained on the portrait genre and "
          "remains blind to it (6.7% caught); averaging diluted CLIP's 93.3% on that genre to 66.7%. The maximum keeps either "
          "model's detection, at a measured cost in false alarms on ordinary photographs (Chapter 5)."),

    ("h3", "Video Pipeline"),
    ("p", "Figure 3.3 shows how a video is parsed. The pipeline was verified line by line against the original Celeb-DF v2 "
          "evaluation script of the model, and a regression test guards that parity."),
    ("fig", "pipe_video", "Video detection pipeline", 6.2),
    ("num", [
        "Frame sampling: sixteen frame indices are chosen with numpy linspace over the whole clip and decoded with OpenCV, "
        "skipping unneeded frames with grab(). Clips shorter than sixteen frames are rejected with a clear error.",
        "Face crop: in each frame a Haar cascade {violajones} finds the largest frontal face, the box is enlarged by 20%, and "
        "the crop is resized to 224×224 with PIL bilinear interpolation and ImageNet normalisation. If no face is found, a "
        "centre square is used; the system records how many of the sixteen frames contained a detected face and warns when "
        "fewer than half did.",
        "Scoring: an EfficientNet-B0 {efficientnet} (Video Model v1, round 4) outputs one fake logit z_i per frame in one "
        "batched forward pass; the clip logit is their mean.",
        "Calibration: the mean logit is mapped through Platt scaling, Equation (3.2), fitted on the round-4 validation split "
        "(926 videos, a = 1.8237, b = −0.0946).",
        "Resolution guard: if the short side of the video is under 360 px the verdict is withheld as UNCERTAIN, because "
        "accuracy collapses on small faces (Chapter 5).",
        "Verdict: the calibrated probability is banded as in Equation (3.3).",
    ]),
    ("eq", [("t", "p̂=σ(a·"), ("t", "z̄"), ("t", "+b),   "), ("t", "z̄"), ("t", "="), ("f", [("t", "1")], [("t", "16")]), ("nary", "∑", "i=1", "16", [("sub", "z", "i")])], "3.2"),
    ("p", "Round 4 was fine-tuned from the original epoch-11 checkpoint on Kaggle with production-identical face crops, "
          "real-world degradation augmentation (JPEG re-compression, down-scaling, blur, noise), four epochs, a learning rate "
          "of 4×10⁻⁵, batch size 64, eight frames per video (sixteen for the small phone sources) and at most 600 crops per "
          "class per dataset per epoch so that no single source dominates; the best epoch was selected by validation AUC."),

    ("h3", "Audio Pipeline"),
    ("p", "Figure 3.4 shows how audio is parsed. Speech is decoded with PyAV to 16 kHz mono and RMS-normalised once over the "
          "whole clip to −20 dBFS, exactly as in training. Up to six 4-second windows (64,000 samples) are taken at evenly "
          "spaced offsets; clips shorter than one window are tiled to length."),
    ("fig", "pipe_audio", "Audio detection pipeline", 6.2),
    ("p", "Each window is passed through a wav2vec 2.0 base model {wav2vec2} whose convolutional feature extractor is "
          "frozen; the 768-dimensional hidden states are mean-pooled and a small head (768 → 256 → 2) gives the deepfake "
          "probability by softmax. The clip score is the mean over windows and the standard deviation is reported alongside. "
          "The model was fine-tuned in successive rounds (v2 to v9) on voice-conversion attacks from ASVspoof 2019 "
          "{asvspoof2019}, DEEP-VOICE {deepvoice}, In-the-Wild {inthewild} split by speaker, genuine LibriSpeech "
          "{librispeech} narration split by speaker (added to remove false alarms on clean studio audio) and VCC 2020 "
          "{vcc2020} voice-conversion systems split by system. Round 9 used AdamW, a learning rate of 10⁻⁵, three epochs, "
          "batch size 8 and class-weighted cross-entropy. The pretrained AASIST {aasist} ONNX model remains selectable as a "
          "fallback."),

    ("h3", "Verdict Bands and Calibrated Confidence"),
    ("p", "All detectors map their score p to a three-way verdict around a threshold t with a margin of 0.2 (Figure 3.5). "
          "For images and audio t = 0.5 is applied to the raw probability; for video it is applied to the Platt-calibrated "
          "probability:"),
    ("fig", "band", "Verdict bands around the decision threshold", 5.2),
    ("eq", [("t", "v(p)=FAKE if p ≥ t+0.2;   REAL if p ≤ t−0.2;   UNCERTAIN otherwise")], "3.3"),
    ("p", "The UNCERTAIN band reflects a deliberate choice that a borderline score is reported as borderline instead of "
          "being forced into a binary answer. Confidence is stored as the probability of the verdict given, max(p, 1−p), and "
          "a separately calibrated confidence is shown next to it. For images it uses temperature scaling {guo2017}, "
          "Equation (3.4), with T = 4.45 fitted on 6,500 labelled evaluation scores; no audio calibration is shown because no "
          "labelled audio evaluation set exists in the harness yet, and the interface says \"not available\" rather than "
          "guessing."),
    ("eq", [("t", "p̂=σ("), ("f", [("t", "logit(p)")], [("t", "T")]), ("t", "),   σ(x)="), ("f", [("t", "1")], [("t", "1+"), ("sup", "e", "−x")])], "3.4"),

    ("h3", "Explanation Layer"),
    ("p", "Explanations are recomputed on demand with the deployed models and verified to reproduce the production numbers "
          "exactly. For images, Grad-CAM {gradcam} is computed on ConvNeXt's last stage with respect to the fake logit. "
          "Because a normalised Grad-CAM map always shows some bright region, even on images the model considers real, the "
          "map's intensity is scaled by ConvNeXt's own fake probability, so a confident REAL verdict shows a near-empty map. "
          "CLIP and the face check have no spatial map; when either drives the verdict, the interface says so instead of "
          "presenting ConvNeXt's map as the explanation. For video, the user sees each frame's probability, the exact crops "
          "the model received and Grad-CAM on the three most suspicious frames. For audio, Grad-CAM does not apply because "
          "wav2vec 2.0 has no late two-dimensional feature map, so input-gradient saliency {simonyan2014} of the deepfake "
          "probability with respect to the waveform is pooled into 80 time bins and overlaid on the spectrogram of the most "
          "suspicious window."),

    ("h3", "Provenance Layer"),
    ("p", "The provenance service (Figure 3.6) runs independently of the detectors and reports four kinds of evidence:"),
    ("fig", "provenance", "Provenance checks and how they reach the user", 5.8),
    ("bul", [
        "**C2PA Content Credentials** {c2pa}, validated with the official SDK against the published trust list: whether the "
        "signature is valid, whether the file is unchanged since signing, whether the signer is trusted, and whether the "
        "manifest declares AI generation through the IPTC digital-source-type field.",
        "**Embedded metadata**: camera EXIF, editing software, the presence (never the coordinates) of GPS data, XMP AI "
        "source type, and generation parameters (prompt, seed, sampler) left by Stable Diffusion front-ends such as "
        "AUTOMATIC1111 and ComfyUI.",
        "**Stable Diffusion / SDXL invisible watermark** {invisiblewm}: the fixed 48-bit message embedded by the reference "
        "pipeline is decoded with a re-implementation (Haar DWT in PyWavelets) verified bit-for-bit against the reference "
        "library; a match requires all 48 bits, and for video eight sampled frames are checked.",
        "**Error level analysis** for JPEG files, shown as a visual aid only and never scored.",
    ]),
    ("p", "Provenance never changes the stored verdict, because metadata can be removed by any re-save. A present AI "
          "declaration is strong evidence; an absent one says nothing either way. When the models say REAL or UNCERTAIN but "
          "the file declares itself AI-generated, the API, the interface and the PDF all show an explicit conflict note, and "
          "the interface switches the headline to the warning style."),

    ("h3", "ProctorLens: Identity Verification and Exam Monitoring"),
    ("p", "ProctorLens reuses the face-analysis stack for proctored online examinations (Figure 3.7). An organisation "
          "account creates MCQ exams and shares a private join code; a student links to the organisation, opens an exam, "
          "and must pass a fresh identity check for every attempt: a government-ID photo is checked for exactly one sharp "
          "face, its FaceNet {facenet} embedding is used transiently, and a live selfie must match it."),
    ("fig", "proctor_flow", "ProctorLens examination flow", 6.2),
    ("p", "Identity is decided by the cosine similarity of 512-dimensional InceptionResNet-v1 (VGGFace2) embeddings of MTCNN "
          "{mtcnn} aligned faces, Equation (3.5), banded around a threshold of 0.44 with a margin of 0.1. The threshold was "
          "calibrated on 1,000 labelled LFW {lfw} pairs as the accuracy-maximising point."),
    ("eq", [("t", "s(u,v)="), ("f", [("t", "u·v")], [("t", "‖u‖ ‖v‖")]), ("t", ",   match if s ≥ 0.44+0.1,   mismatch if s ≤ 0.44−0.1")], "3.5"),
    ("p", "During the attempt the browser sends a webcam frame (and optionally a short audio clip) every 5–20 seconds as a "
          "background check job. Each check computes eight independent signals (Figure 3.8); a check is flagged if any signal "
          "fires, even when identity matches. Browser events (tab or focus loss, clipboard paste, a devtools heuristic, camera "
          "loss) are recorded separately. Enforcement is server-side: in the same database transaction as the check that "
          "crosses the exam's max_flags limit, the attempt is marked terminated_for_flags, so a modified client cannot keep "
          "submitting answers."),
    ("fig", "monitor_signals", "Signals computed for every monitoring check", 6.2),
    ("p", "Head pose is estimated from MTCNN's five landmarks with OpenCV's EPnP solver and compared with the session's own "
          "first good reading rather than a universal angle, because no labelled head-pose data was available to validate "
          "absolute angles. The liveness signal is not blink detection (five landmarks have no eyelid points); it flags a "
          "face whose landmarks stay almost perfectly still across three consecutive checks, as a printed photo would. "
          "Talking is detected by short-term-energy voice activity with an adaptive noise floor; it signals speech, not whose "
          "voice it is. These monitoring thresholds are stated placeholders, documented as such in the code."),

    ("h3", "Performance Metrics"),
    ("p", "The accuracy of a classifier is computed using Equation (3.6):"),
    ("eq", [("t", "Accuracy="), ("f", [("t", "TP+TN")], [("t", "TP+TN+FP+FN")])], "3.6"),
    ("p", "where TP, TN, FP and FN denote true positives, true negatives, false positives and false negatives "
          "respectively, with \"fake\" as the positive class. We also report fake recall (sensitivity), real specificity, "
          "precision and F1-score, Equations (3.7)–(3.9), ROC-AUC with 95% bootstrap confidence intervals (1,000 resamples), "
          "and the expected calibration error over ten equal-width bins, Equation (3.10). The metric code was verified "
          "against scikit-learn to within 2×10⁻¹⁶."),
    ("eq", [("t", "Recall="), ("f", [("t", "TP")], [("t", "TP+FN")]), ("t", ",   Specificity="), ("f", [("t", "TN")], [("t", "TN+FP")])], "3.7"),
    ("eq", [("t", "Precision="), ("f", [("t", "TP")], [("t", "TP+FP")])], "3.8"),
    ("eq", [("t", "F1="), ("f", [("t", "2·Precision·Recall")], [("t", "Precision+Recall")])], "3.9"),
    ("eq", [("t", "ECE="), ("nary", "∑", "b=1", "10", [("f", [("sub", "n", "b")], [("t", "n")]), ("t", "|acc(b)−conf(b)|")])], "3.10"),

    ("h2", "Tools and Technologies Used"),
    ("bul", [
        "**Programming languages:** Python 3.11 (backend, models, evaluation), JavaScript/JSX (frontend).",
        "**Backend:** FastAPI with uvicorn, Pydantic, async SQLAlchemy over SQLite (aiosqlite), python-jose (HS256 JWT), "
        "passlib/bcrypt, ReportLab (PDF), c2pa-python, PyWavelets, PyAV, Pillow + pillow-heif.",
        "**Machine learning:** PyTorch 2.8 (CPU) and torchvision, OpenCLIP, Hugging Face Transformers (wav2vec 2.0), "
        "facenet-pytorch (MTCNN, InceptionResNet-v1), ONNX Runtime (AASIST, YOLOv8n), OpenCV, NumPy, scikit-learn.",
        "**Frontend:** React 18, React Router 7, Vite 5, Tailwind CSS 3, Axios, GSAP and Framer Motion (animation), "
        "MediaPipe Tasks Vision (live face-mesh overlay), Lucide icons.",
        "**Training:** Kaggle notebooks and Google Colab (free cloud GPUs) for fine-tuning; local CPU for evaluation.",
        "**DevOps:** Docker and Docker Compose (development and production), nginx 1.27, GitHub and GitHub Actions CI, "
        "pytest.",
        "**Hardware:** development laptops (Windows 11, CPU inference); deployment target is a single CPU server, about "
        "265 MB resident memory after all image and video models are loaded.",
    ]),

    # =========================================================================================== 4
    ("h1", "System Design and Implementation"),
    ("p", "This chapter turns the methodology into a working system: the requirements, the UML and data-flow views, the "
          "database and API design, the user interface, security, testing and deployment."),

    ("h2", "Requirements"),
    ("h3", "Functional Requirements"),
    ("bul", [
        "FR1: Users can sign up as a student or an organisation, log in and log out; sessions use 24-hour JWTs.",
        "FR2: Users can upload an image, video or audio file and receive a verdict, confidence, calibrated confidence and "
        "per-model sub-scores.",
        "FR3: Long video scans run in the background with real progress, queue position and cancel.",
        "FR4: Users can open an explanation and a provenance panel for any of their results and download a PDF report.",
        "FR5: Users can view their history and dashboard statistics, give \"was this correct?\" feedback with opt-in reuse "
        "consent, and delete their account with all files and reports.",
        "FR6: Organisations can create exams and questions, share a join code, manage the roster and review attempts, flag "
        "timelines and session reports.",
        "FR7: Students can join an organisation, verify their identity against a government ID and take a proctored exam "
        "that is scored on completion or terminated when the flag limit is reached.",
    ]),
    ("h3", "Non-Functional Requirements"),
    ("bul", [
        "**Privacy:** every upload, result, job, report, feedback item and session is visible only to its owner; others "
        "receive 404, never 403, so an identifier does not reveal that a resource exists. GPS coordinates are never "
        "returned. ID embeddings are used transiently.",
        "**Honesty:** no fabricated numbers in the UI; borderline scores are UNCERTAIN; missing metadata is never presented "
        "as evidence of authenticity; every model claim carries its measured limits.",
        "**Responsiveness:** image and audio scans complete within the request (about 0.5–1.2 s and 3.5 s on CPU in the "
        "demonstrations in Chapter 5); video scans never block other users.",
        "**Security:** bcrypt password hashing, signed tokens, magic-byte upload validation, rate limits, CSP and refusal to "
        "start in production with a default secret.",
        "**Maintainability:** swappable model back-ends selected by environment variables, documented rollback paths, and "
        "an automated test suite run in CI.",
    ]),

    ("h2", "Use-Case Diagram"),
    ("p", "Figure 4.1 shows the actors and use cases. A user (or student) performs media verification and takes proctored "
          "exams; an organisation administers exams and reviews attempts; the periodic monitoring check is triggered by the "
          "exam page on the student's behalf."),
    ("fig", "usecase", "Use-case diagram", 6.0),

    ("h2", "Data Flow Diagrams"),
    ("p", "Figure 4.2 is the context-level (level 0) DFD and Figure 4.3 the level 1 DFD, which decomposes the platform into "
          "seven processes and four data stores."),
    ("fig", "dfd0", "Data flow diagram, level 0", 5.6),
    ("fig", "dfd1", "Data flow diagram, level 1", 6.0),

    ("h2", "Sequence Diagrams"),
    ("p", "Figure 4.4 shows a synchronous scan of an image or audio file followed by the optional evidence and report "
          "requests. POST /detect is idempotent: an upload has at most one result. (A repeated call once stored a second row "
          "and broke every read endpoint for that upload; a regression test now guards against it.) Figure 4.5 shows a video "
          "scan as a background job."),
    ("fig", "seq_scan", "Sequence diagram: image or audio scan and evidence requests", 6.0),
    ("fig", "seq_job", "Sequence diagram: background video scan", 6.0),
    ("p", "Progress is real, not a timer: frame decoding reports 0–70%, face cropping 70–90% and scoring the rest. The "
          "callbacks are observation-only, and predictions were verified bit-identical with and without them. Cancelling "
          "raises an exception inside the callback, so the scan stops and nothing is stored. At most "
          "DETECTION_JOB_CONCURRENCY jobs run at once (default one, since inference is CPU-bound)."),

    ("h2", "Database Design"),
    ("p", "Figure 4.6 shows the entity-relationship diagram. A detection result belongs to exactly one upload and has one "
          "modality-specific analysis row. Feedback is unique per (upload, user). The audit log deliberately has no foreign "
          "key to users, so an \"account deleted\" entry survives the account it describes. An exam attempt is linked one to "
          "one with the monitoring session that proctors it. New nullable columns are added by light, idempotent migrations "
          "at start-up."),
    ("fig", "er", "Entity-relationship diagram", 6.2),

    ("h2", "API Design"),
    ("p", "All routes are mounted under /api/v1 and documented automatically by FastAPI's OpenAPI page. Table 4.1 lists the "
          "main endpoints."),
    ("tab", "Main REST endpoints",
     ["Method and path", "Purpose"],
     [["POST /auth/signup, /auth/login; GET, DELETE /auth/me", "Account creation, login (JWT), profile, full account deletion"],
      ["POST /upload/; GET /upload/{id}", "Validated upload (size, MIME, magic bytes); upload metadata"],
      ["POST /detect/{id}; GET /detect/{id}/result", "Synchronous detection (idempotent); stored result"],
      ["POST /detect/{id}/jobs; GET, DELETE /jobs/{job}", "Background video scan with progress; poll or cancel"],
      ["GET /detect/{id}/explain; /provenance", "Grad-CAM, per-frame or audio saliency; C2PA, metadata, watermark, ELA"],
      ["GET, PUT, DELETE /detect/{id}/feedback", "\"Was this correct?\" feedback with consent; withdraw"],
      ["GET /report/{id}; /history; /stats", "PDF report; history list; dashboard counts"],
      ["POST /identity/verify-government-id; /match", "Government-ID enrolment; one-shot identity match"],
      ["POST /identity/sessions/{id}/checks[/jobs]; /events", "Monitoring checks (sync or job); browser events"],
      ["POST /exams; /exams/{id}/questions; GET /exams/mine", "Exam and question management; exams for a student"],
      ["POST /exams/{id}/attempts/verify-identity; /attempts", "Per-attempt identity check; start attempt"],
      ["POST /exams/attempts/{id}/answers; /finish", "Record answers; score and close the attempt"],
      ["GET /admin/join-code, /roster, /sessions, /exams/{id}/attempts", "Organisation administration and review"],
      ["GET /health; /health/ready", "Liveness; readiness (database + every checkpoint present, else 503)"]],
     [4300, 4366]),

    ("h2", "Frontend Implementation"),
    ("p", "The client is a React single-page application. App.jsx defines the live routes (home, verify, history, reports, "
          "report detail, dashboard, ProctorLens, exam attempt and result, admin, about, login and sign-up), protected by "
          "ProtectedRoute and GuestRoute wrappers; every page is lazily loaded, which reduced the main bundle from 566 KB to "
          "447 KB. A single Axios client attaches the bearer token to every request, converts errors into friendly messages "
          "that include the server's request ID, and redirects to a clear \"session expired\" message on a 401. The visual "
          "language is a \"forensic evidence bench\" (dark ground, bone text, brass accents, serif headings). Key components "
          "are UploadFlow (the scan flow), ExplanationPanel (lazy, fetched only when opened), ProvenancePanel, FeedbackPanel, "
          "LowResolutionNote and the ProctorLens components (GovernmentIdCapture, WebcamCapture and a live face-mesh overlay)."),

    ("h2", "PDF Report Generation"),
    ("p", "The report generator (ReportLab) produces a forensic report for every result: a verdict banner written for a "
          "non-expert, a file and confidence table, technical details (model, timing, per-model probabilities, frames with a "
          "detected face, resolution), a scope note for video, an explainability section with the same images as the "
          "interface, and a provenance section. Explainability and provenance are best-effort: a failure in either only "
          "skips that section. A separate session report summarises a proctored attempt."),

    ("h2", "Security and Privacy"),
    ("bul", [
        "Passwords are hashed with bcrypt; tokens are HS256 JWTs with a 24-hour expiry. In production mode the server "
        "refuses to start if the signing key is the public development default or shorter than 32 characters.",
        "Uploads are validated by declared MIME type and by their leading bytes, so a renamed executable is rejected. HEIC "
        "images are supported end to end.",
        "Ownership is checked on every route and answered with 404 for other users' data.",
        "The production nginx configuration adds a Content-Security-Policy derived from auditing the built bundle, security "
        "headers, and per-IP rate limits of 10 requests per minute on authentication and 20 per minute on upload and "
        "detection.",
        "Security-relevant actions (sign-up, login, failed login, feedback withdrawal, account deletion, feedback export) "
        "are written to an audit log; every response carries an X-Request-ID that is echoed in error messages.",
        "Feedback files are kept for evaluation only with explicit, per-item, withdrawable consent (off by default). Users "
        "can delete their account, which removes every upload, report, result and feedback row.",
    ]),

    ("h2", "Testing"),
    ("p", "The backend has a pytest suite of 46 test modules: 512 tests pass and 18 skip when optional checkpoints are "
          "absent. The CI subset (pytest -m \"not models\") needs no model files and runs on GitHub Actions together with a "
          "clean frontend build. The suite covers authentication, upload validation, 404-not-403 ownership on every route, "
          "error mapping, verdict banding including the protected thresholds, calibration, jobs (non-blocking, cancel, "
          "queue position), explainability and provenance logic (tamper, strip, hostile metadata), the watermark decoder, "
          "PDF content, feedback consent, deployment helpers, identity, monitoring, exams and the evaluation metrics against "
          "scikit-learn. Checkpoint-dependent tests reproduce production scores exactly. A mutation script deliberately "
          "breaks 17 guarded behaviours (for example the thresholds, ownership checks, idempotent detect and consent gating) "
          "and confirms that the named test fails each time."),

    ("h2", "Deployment"),
    ("p", "Figure 4.7 shows the production deployment. docker-compose.prod.yml runs a frontend container (nginx serving the "
          "built SPA and proxying /api) and a backend container (non-root, a single uvicorn process without reload, "
          "DEBUG=false). State lives on a named volume; model checkpoints are mounted read-only and verified against SHA-256 "
          "checksums by a fetch script that discards any mismatched download. A readiness probe returns 503 until the "
          "database and every checkpoint are available."),
    ("fig", "deploy", "Production deployment", 6.0),

    ("h2", "Project Timeline"),
    ("p", "Figure 4.8 shows the implementation timeline reconstructed from the repository history (47 commits by the team); "
          "work before the first commit, such as the literature survey and the first CIFAKE model, is shown hatched because "
          "its exact dates were not recorded."),
    ("fig", "gantt", "Implementation timeline", 6.2),

    # =========================================================================================== 5
    ("h1", "Results and Discussion"),
    ("p", "This chapter presents the measured performance of the deployed system and screenshots of it at work. All "
          "measurements use the production detector classes with unchanged weights and thresholds, and every data source "
          "is tagged seen or unseen so that fit and generalisation are never mixed."),

    ("h2", "Experimental Setup"),
    ("p", "Evaluation ran on CPU on the team's Windows 11 development machines using the harness in backend/eval "
          "(build_manifest → run_predictions → make_report). Training and fine-tuning ran on Kaggle and Colab GPUs. Image "
          "evaluation used 6,532 files from twelve sources (up to 500 per class per dataset); video used 300 unseen RTFS "
          "clips plus ten team videos for the epoch-11 model, and 356 held-out videos for the round-4 comparison. "
          "Robustness used 300 label-balanced images (180 seen, 120 unseen) under 21 settings, following the corruption "
          "methodology of {hendrycks2019}, and 30 RTFS clips under 11 settings for video. Identity used 1,000 LFW pairs; "
          "fairness used 600 UTKFace {utkface} portraits, 60 per gender-race group."),

    ("h2", "Results"),
    ("h3", "Image Detection: Seen Versus Unseen Sources"),
    ("p", "Figure 5.1 and Table 5.1 contain the central result of this work. On seen sources every configuration is "
          "near perfect. On unseen generators the deployed ensemble catches only 27.3% of fakes, and AUC drops to 0.665."),
    ("fig", "c_image_seen_unseen", "Image detection by training status of the source (threshold 0.5)", 6.2),
    ("tab", "Image detection by training status of the source",
     ["System", "Data", "Accuracy (%)", "Fake recall (%)", "Real specificity (%)", "AUC"],
     [["ConvNeXt", "Seen", "99.3", "99.5", "99.2", "1.000"], ["CLIP", "Seen", "97.6", "97.8", "97.5", "0.997"],
      ["Mean of both", "Seen", "99.6", "99.8", "99.4", "1.000"], ["Max (deployed)", "Seen", "98.4", "100.0", "96.8", "1.000"],
      ["ConvNeXt", "Unseen", "50.2", "12.0", "98.7", "0.665"], ["CLIP", "Unseen", "49.7", "20.6", "86.7", "0.640"],
      ["Mean of both", "Unseen", "48.5", "10.9", "96.3", "0.670"], ["Max (deployed)", "Unseen", "53.2", "27.3", "86.0", "0.665"]],
     [1900, 1100, 1400, 1500, 1700, 1066]),
    ("p", "Per source (Figure 5.2), the deployed system catches 30.7% of GenImage Stable Diffusion images, 20.0% of "
          "Midjourney and 26.0% of BigGAN images, and reaches 54.3% accuracy on OpenFake. All eight photorealistic ChatGPT "
          "\"everyday\" images were labelled REAL with more than 99.99% confidence. The maximum rule trades specificity for "
          "recall: compared with the mean it raises unseen-fake recall from 10.9% to 27.3% but lowers real-photo specificity "
          "from 96.3% to 86.0%; on the portrait-app genre it keeps CLIP's 93.3%. Figure 5.3 shows the ROC and "
          "precision-recall curves over all 6,532 images."),
    ("fig", "c_unseen_sources", "Share of AI images caught per unseen source (deployed max ensemble)", 5.6),
    ("fig", "@roc_image", "ROC and precision-recall curves of the image sub-models and ensembles (6,532 images)", 6.2),

    ("h3", "Face-Swap Check in the Image Pipeline"),
    ("p", "Figure 5.4 compares the face-swap check with the whole-image models. On frames from 356 held-out research "
          "videos the face check reaches AUC 0.95 against 0.61 for the image ensemble, and at a 5% false-alarm budget it "
          "catches 78% of fake frames against 12%. On inswapper swaps of 80 CelebA photos it reaches 0.859 against 0.659. "
          "On RWFS, 500 swaps from eight real online face-swap apps, the first version was near chance (0.617), which "
          "prompted photo model round 2: in a leave-one-group-out test its held-out-group AUC rose to 0.77–0.83 while keeping "
          "the research-frame AUC at 0.931 and the share of real faces wrongly flagged at 4.9%. These figures were measured on "
          "research datasets; no photos from the specific consumer apps users may encounter were available for testing."),
    ("fig", "c_faceswap", "Face-swap check versus whole-image models (ROC-AUC)", 6.0),

    ("h3", "Video Detection"),
    ("p", "With the original epoch-11 weights, the video model reached 79.4% accuracy and AUC 0.892 [0.852, 0.926] on 310 "
          "unseen clips (RTFS plus ten team videos), catching 77.3% of inswapper and 72.0% of uniface swaps while keeping "
          "84.0% of real originals real. Inspecting the face crops explained the two consumer-app misses: in the Akool clip "
          "the Haar cascade locks onto a wall poster and a wrist, and the Magic Hour clip is stored rotated by 90° with no "
          "rotation flag, so no face is found in any frame. Three preprocessing fixes were tested on all 300 RTFS clips "
          "(Figure 5.5): a rotation fallback that only activates when no frame has a face (safe, but it never activated on "
          "RTFS), a two-cascade consensus (removes the poster detections but also good ones, AUC 0.890) and wider 35% padding "
          "(AUC 0.849, 5.3 points lower specificity). The existing crop geometry is close to a local optimum, so further "
          "gains needed retraining."),
    ("fig", "c_video_variants", "Video preprocessing variants on 300 unseen RTFS clips", 6.0),
    ("p", "The raw scores of the epoch-11 model sat almost entirely between 0.2 and 0.7, so the original ±0.2 band around "
          "0.525 made 294 of 300 RTFS videos UNCERTAIN. Platt rescaling of the mean logit, fitted on a grouped split, raised "
          "the share of decided RTFS test videos from 1.7% to 75% with 91.1% accuracy among them. Figure 5.6 shows why the "
          "resolution guard was added: AUC falls from 0.923 at 1080p to 0.729 at 240 px, and FakeAVCeleb's 224 px clips are "
          "at chance; without the guard the rescaling produced confident wrong verdicts on such clips."),
    ("fig", "c_video_res", "Video ROC-AUC against the short side of the video", 4.6),
    ("p", "Round 4 fine-tuning (Figure 5.7) raised the AUC on 356 local held-out videos from 0.765 to 0.961, reduced real "
          "videos wrongly flagged FAKE from 21 to 8 of 185 and increased fakes caught from 82 to 114 of 171; on held-out DFD "
          "and phone actors the AUC rose from 0.845 to 0.977. Fully AI-generated video (five Sora/Kling clips) is out of "
          "scope: none were caught, and the PDF report says so."),
    ("fig", "c_video_r4", "Video Model v1: epoch 11 versus round 4 on held-out videos", 6.2),

    ("h3", "Audio Detection"),
    ("p", "The audio model was assessed on the team's six held-out acceptance sets during fine-tuning and passed five of "
          "the six, including 10 of 10 genuine clips from previously unseen LibriSpeech speakers (Figure 5.8). On a stress "
          "test with one clip from each of the 59 VCC 2020 voice-conversion systems, the previous model version scored 30/59 "
          "(50.8%, essentially chance); adding a system-split VCC source raised this to 98.3%. Because about 40 of those "
          "systems were used in training, the figure is partly a measure of fit. These results come from the training "
          "notebooks and have not yet been reproduced in the evaluation harness, and audio robustness to compression and "
          "noise has not been measured, because no labelled audio dataset was available locally."),
    ("fig", "c_audio", "Audio model results on held-out acceptance sets", 4.8),

    ("h3", "Robustness to Everyday Degradation"),
    ("p", "Common sharing pipelines leave the image ensemble almost unaffected on seen sources: screenshot, WhatsApp-style "
          "(longest side 1600 px, JPEG quality 65) and WebP processing all keep AUC at 0.99 or above. Figure 5.9 shows the "
          "exceptions. CLIP raises false alarms on real photographs under faint noise and compression: the ensemble's "
          "real-photo specificity falls from 95.6% clean to 87.8% at JPEG quality 90, 71.1% at noise σ = 5 and 62.2% at "
          "quality 10, while ConvNeXt alone stays between 96.7% and 98.9% in all 21 settings. Under heavy blur (σ = 3) the "
          "opposite happens: ConvNeXt catches only 50% of fakes and the ensemble recovers to 83.3% because CLIP still detects "
          "them. The maximum rule therefore inherits CLIP's false alarms and ConvNeXt's misses under different degradations."),
    ("fig", "c_img_robust", "Image robustness on seen sources: specificity and recall under degradation", 6.2),
    ("p", "Video was far more fragile (Figure 5.10, epoch-11 weights, 30 clips). At JPEG quality 10 every degraded fake was "
          "labelled REAL; blur σ = 3 and 4× down-scaling pushed AUC below 0.5, meaning the score pointed the wrong way; "
          "WhatsApp-style processing alone dropped AUC to 0.61. This finding directly motivated the degradation augmentation "
          "used in round 4; the robustness study has not yet been repeated for round 4."),
    ("fig", "c_vid_robust", "Video robustness with every sampled frame degraded (30 RTFS clips)", 6.0),

    ("h3", "Explanations, Provenance and Calibration"),
    ("p", "**Faithfulness.** For 195 fakes that ConvNeXt scores FAKE, masking the 20% of the image that Grad-CAM ranks "
          "highest lowered the fake probability by 0.232 on average, against 0.044 for a random mask of the same area; the "
          "Grad-CAM mask had the larger effect on 85% of images, so the maps do point at regions the score depends on."),
    ("p", "**Provenance.** Across the 6,532 evaluation images only 0.4% of AI images carry any AI declaration, because "
          "benchmark datasets are re-encoded and stripped. The signal is valuable exactly where the detectors fail, however: "
          "all eight ChatGPT images carry valid, unmodified OpenAI Content Credentials declaring AI generation, and in total "
          "9 of the 1,026 AI images the ensemble misses declare themselves AI-generated. The declaration survived a "
          "byte-for-byte copy in 8 of 8 cases and every re-encode we tried in 0 of 8. ELA statistics separate real from fake "
          "with AUC 0.53–0.75 on most sources but 0.995 on DeepDetect, which reveals a JPEG-history shortcut in that dataset."),
    ("p", "**Calibration.** Both detectors are mis-calibrated in raw form (Figure 5.11). Temperature scaling lowered the "
          "image ECE from 0.158 to 0.056 and the video ECE from 0.254 to 0.059; the video Platt fit lowered pooled ECE on a "
          "separate 161-video test split from 0.161 to 0.078. The temperature-scaled values were measured on the fitting "
          "scores and are therefore optimistic."),
    ("fig", "c_calibration", "Expected calibration error before and after calibration", 4.8),

    ("h3", "Fairness Across Demographic Groups"),
    ("p", "To check whether genuine photos of some groups are wrongly flagged more often, 600 real UTKFace portraits were "
          "scored, 60 per gender-race group (Figure 5.12). The deployed ensemble's specificity is 94.8% overall but ranges "
          "from 90.0% to 100% across groups; ConvNeXt alone spans 96.7–100% and CLIP alone 91.7–100%. CLIP's two weakest "
          "groups are exactly the ensemble's two weakest, pointing again to CLIP as the source of false alarms. With 60 images "
          "per group the ranking of specific groups is noisy; the size of the spread and the sub-model it comes from are the "
          "reliable findings."),
    ("fig", "c_fairness", "Real-photo specificity per gender and race group (600 UTKFace portraits)", 6.2),

    ("h3", "Identity Verification"),
    ("p", "On 1,000 LFW DevTest pairs scored with the deployed pipeline, same-person similarity averaged 0.720 (σ 0.201) and "
          "different-person similarity 0.029 (σ 0.160), with ROC-AUC 0.974. At the chosen threshold of 0.44 the accuracy is "
          "96.5% with a 0.4% false-accept rate and a 6.6% false-reject rate (Figures 5.13 and 5.14). LFW consists of frontal, "
          "well-lit press photographs; webcam enrolment under poor lighting will plausibly overlap more, so this is a "
          "data-grounded starting point, not a guaranteed production error rate."),
    ("fig", "@identity_hist", "Distribution of cosine similarity for same-person and different-person LFW pairs", 5.0),
    ("fig", "c_identity", "Identity matching at threshold 0.44 on LFW", 4.6),

    ("h3", "The Working System"),
    ("p", "Figures 5.15 to 5.24 show the deployed application processing real files through the live API on CPU, captured "
          "on the final build. The home page (Figure 5.15) introduces the examination flow; the verify page (Figure 5.16) "
          "accepts all three media types; history and dashboard (Figures 5.17 and 5.18) show only the signed-in user's scans."),
    ("fig", "@home", "Home page", 6.0),
    ("fig", "@verify", "Verify page: upload of an image, video or audio file", 6.0),
    ("fig", "@history", "Scan history (owner-only)", 6.0),
    ("fig", "@dashboard", "Dashboard built from real counts only", 6.0),
    ("p", "Figure 5.19 shows a photorealistic ChatGPT image, one of the documented blind spots. The models call it REAL, but "
          "its C2PA credentials declare AI generation, so the interface shows the provenance conflict and switches the "
          "headline to the warning style; Grad-CAM is faded because ConvNeXt's own fake probability is low and the check "
          "that drove the score has no map. Figure 5.20 shows a portrait-app image whose stored verdict is UNCERTAIN (58%); "
          "its embedded metadata declares AI generation, so the same conflict display applies."),
    ("fig", "@chatgpt_top", "Result for a ChatGPT-generated photograph: REAL by the models, AI-declared by provenance", 5.4),
    ("fig", "@portrait_top", "Result for a portrait-app image: UNCERTAIN by the models, AI-declared by metadata", 5.4),
    ("p", "Figure 5.21 shows a face-swap video (an RTFS inswapper clip; RTFS was part of round-4 training, so this is a "
          "demonstration, not evidence of generalisation): the calibrated probability is 99.8%, sixteen of sixteen frames had "
          "a detected face, and the per-frame bars, the exact crops and Grad-CAM on the three most suspicious frames are "
          "shown. Figure 5.22 shows an audio scan with the per-window score and the time-saliency curve over the spectrogram; "
          "the clip's ground truth is not known, so it illustrates the output only."),
    ("fig", "@video_explain", "Video result: per-frame scores, face crops and Grad-CAM on the most suspicious frames", 5.4),
    ("fig", "@audio_explain", "Audio result: per-window score and input-gradient saliency over the spectrogram", 5.4),
    ("fig", "@pdf_video", "First page of the generated PDF forensic report for the video", 4.6),
    ("fig", "@proctorlens", "ProctorLens student hub: joining an organisation to see its exams", 6.0),

    ("h2", "Comparison with Existing Methods"),
    ("p", "Published detectors are evaluated on different data, so their headline numbers cannot be compared directly with "
          "ours. Table 5.2 therefore compares the configurations we built and measured on the same 6,532 images (all sources "
          "pooled, threshold 0.5), together with the video model on 310 unseen clips. The deployed maximum rule has the best "
          "pooled accuracy, recall and F1, at the cost of precision; ConvNeXt alone is the most precise."),
    ("tab", "Performance comparison of the evaluated methods",
     ["Method", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)", "AUC"],
     [["ConvNeXt-Tiny (end-to-end, cf. {wang2020})", "80.2", "98.6", "63.1", "77.0", "0.901"],
      ["CLIP ViT-B/16 probe (cf. {ojha2023})", "79.1", "91.9", "66.0", "76.8", "0.857"],
      ["Mean of ConvNeXt and CLIP", "79.8", "97.6", "63.0", "76.6", "0.888"],
      ["Proposed: max ensemble (deployed)", "80.9", "91.6", "70.0", "79.4", "0.886"],
      ["Video Model v1, epoch 11 (310 unseen clips)", "79.4", "82.7", "74.2", "78.2", "0.892"]],
     [3200, 1100, 1100, 1100, 1100, 1066]),
    ("p", "Table 5.3 compares TruthLens with typical published detectors and public online checkers by capability. The "
          "contribution is less a new network than a complete, measured system: three modalities, an evidence layer, "
          "explicit uncertainty, honest evaluation and a proctoring application of the same components."),
    ("tab", "Capability comparison with typical existing approaches",
     ["Capability", "Research detectors {ffpp,ojha2023,aasist}", "Typical online checkers", "TruthLens"],
     [["Image, video and audio", "One modality each", "Often one", "All three"],
      ["Explanation of each decision", "Rare", "Rare", "Grad-CAM, per-frame, saliency"],
      ["Provenance (C2PA, metadata, watermark)", "No", "Sometimes", "Yes, with conflict note"],
      ["Explicit UNCERTAIN band", "No", "No", "Yes"],
      ["Calibrated confidence", "Rarely reported", "Not stated", "Temperature / Platt"],
      ["Seen vs unseen evaluation", "Partly", "Not published", "Yes, every source tagged"],
      ["Robustness and fairness measured", "Some", "Not published", "Yes"],
      ["Identity verification and proctoring", "No", "No", "ProctorLens"]],
     [2800, 2200, 1800, 1866]),

    ("h2", "Discussion and Limitations"),
    ("p", "The results support one main lesson: a detector that scores 98% on benchmark-like data can be close to chance on "
          "the next generator, and a user cannot tell which situation applies to their file. For that reason TruthLens "
          "describes REAL as \"no evidence of generation found\" rather than \"authentic\", reserves FAKE for clear scores, "
          "and places provenance and explanations beside the number. The main limitations are:"),
    ("bul", [
        "**Training-data coverage.** Photorealistic GPT-class images and some consumer face-swap apps are absent from the "
        "training sources, and both are confirmed blind spots; fully generated video is out of scope.",
        "**Dataset shortcuts.** DeepDetect's compression history, GenImage's JPEG-versus-PNG split and RTFS's different "
        "encodes may inflate measured accuracy.",
        "**Small samples.** The blind-spot sets (8 and 15 images), the team videos (10), the video robustness subset (30) "
        "and the per-group fairness samples (60) support qualitative conclusions only.",
        "**Not re-measured for the newest models.** The robustness study and the 300-clip RTFS harness used the epoch-11 "
        "video weights; the audio model has not been evaluated in the harness.",
        "**Proctoring thresholds.** Apart from the identity threshold, monitoring thresholds are placeholders without a "
        "labelled validation set, so flags are prompts for human review.",
        "**Engineering.** Job state lives in one server process, and inference is CPU-only in our deployment.",
    ]),
    ("p", "A forensic tool can cause harm in both directions: a false FAKE can discredit a genuine photograph and a false "
          "REAL can lend credibility to a hoax. The verdict bands, explicit conflict notes, \"lead, not proof\" wording and the "
          "absence of any automated action on a verdict (other than exam termination by a limit the organisation sets) are "
          "our mitigations."),

    # =========================================================================================== 6
    ("h1", "Conclusion and Future Scope"),
    ("h2", "Conclusion"),
    ("p", "This project designed, built and evaluated TruthLens, a multimodal platform that detects AI-generated and "
          "manipulated images, videos and speech and presents each verdict together with the evidence behind it. Each media "
          "type has its own pipeline: a ConvNeXt-Tiny, CLIP and face-swap ensemble for images; a face-crop EfficientNet-B0 "
          "with Platt calibration and a resolution guard for video; and a fine-tuned wav2vec 2.0 model for audio. Around "
          "them, an evidence layer adds Grad-CAM and saliency explanations, C2PA, metadata and watermark provenance with "
          "explicit conflict notes, calibrated confidence and PDF reports. ProctorLens extends the same components to "
          "identity-verified, continuously monitored online examinations with server-side enforcement."),
    ("p", "All the objectives set in Chapter 1 were met. The system is a complete, tested and deployable application with "
          "owner-only privacy, upload validation, rate limiting, 512 passing automated tests and a production configuration. "
          "The evaluation was as important as the implementation: it showed near-perfect accuracy on familiar data (98.4%) "
          "and weak generalisation to unseen generators (53.2%), a severe video fragility under compression that guided the "
          "round-4 fine-tune (held-out AUC 0.765 to 0.961), the value of provenance exactly where detectors fail, and a "
          "CLIP-driven spread in false alarms across demographic groups. The central conclusion is that detector scores "
          "should be presented as one strand of evidence, with their uncertainty and limits stated, rather than as a final "
          "answer."),
    ("h2", "Future Scope"),
    ("bul", [
        "Retrain the image models with more diverse generators (GPT-class image models, Flux, recent Midjourney versions) "
        "and with noise and compression augmentation, then revisit the maximum-versus-mean trade-off.",
        "Source fake clips from consumer face-swap apps and select checkpoints on a real-world set that is not also used "
        "for selection; repeat the robustness study for the round-4 video model.",
        "Evaluate the audio model in the harness on a labelled dataset and under MP3, noise and volume degradation, and fit "
        "an audio calibration.",
        "Add a temporal or audio-visual model for lip-sync consistency in talking-head forgeries, and coverage of fully "
        "generated video.",
        "Fit calibration on held-out splits and extend the fairness analysis to video, audio and identity matching.",
        "Validate the proctoring thresholds on labelled webcam data and add a trained anti-spoofing (liveness) model.",
        "Move job state to a shared queue for horizontal scaling and add GPU inference for faster scans.",
    ]),

    # =========================================================================================== 7
    ("h1", "List of Publications"),
    ("h2", "List of Published Papers"),
    ("num", [
        "M. Jaiswal, R. Shinde, A. Pethkar, and S. Thakur, \"TruthLens: An Evidence-Oriented Multimodal Platform for "
        "Detecting AI-Generated Media,\" manuscript prepared in IEEE conference format, 2026. [Venue, status and DOI to be "
        "filled in by the authors.]",
    ]),
    ("h2", "Details of Participation in Project Competitions"),
    ("p", "[To be filled in by the team: name of the competition, organiser, date and outcome, if any.]"),
]

# Extra references not in docs/paper/paper_content.py (IEEE style, plain text).
EXTRA_REFERENCES = {
    "platt1999": "J. Platt, “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” in Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61–74.",
    "mtcnn": "K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint face detection and alignment using multitask cascaded convolutional networks,” IEEE Signal Processing Letters, vol. 23, no. 10, pp. 1499–1503, 2016.",
    "facenet": "F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A unified embedding for face recognition and clustering,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 815–823.",
    "vggface2": "Q. Cao, L. Shen, W. Xie, O. M. Parkhi, and A. Zisserman, “VGGFace2: A dataset for recognising faces across pose and age,” in Proc. IEEE Int. Conf. Automatic Face & Gesture Recognition (FG), 2018, pp. 67–74.",
    "lfw": "G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, “Labeled Faces in the Wild: A database for studying face recognition in unconstrained environments,” Univ. of Massachusetts, Amherst, Tech. Rep. 07-49, 2007.",
    "yolov8": "G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” software, version 8.0, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
}

# Screenshot crops: key -> (file in screens/, (left, top, right, bottom) or None)
SHOTS = {
    "home": ("home.png", None), "verify": ("verify.png", None), "history": ("history.png", None),
    "dashboard": ("dashboard.png", None), "proctorlens": ("proctorlens.png", None),
    "chatgpt_top": ("report_chatgpt_explain.png", (290, 340, 1150, 1700)),
    "portrait_top": ("report_portrait_explain.png", (292, 405, 1145, 1630)),
    "video_explain": ("report_video_explain.png", (285, 410, 1150, 1870)),
    "audio_explain": ("report_audio_explain.png", (290, 345, 1150, 1410)),
    "pdf_video": ("pdf_video_p1.png", None),
}
EXISTING = {
    "roc_image": "backend/eval/results/image/roc_pr.png",
    "identity_hist": "backend/eval/results/identity/similarity_distributions.png",
}
