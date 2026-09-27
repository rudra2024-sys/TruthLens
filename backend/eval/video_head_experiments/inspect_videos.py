import cv2
import os
import numpy as np

out_dir = r"C:\Users\admin\AppData\Local\Temp\claude\C--TrueLense-tl\0c6ba6b3-3bab-4ca8-b50b-fd14db0f746c\scratchpad\frames"
os.makedirs(out_dir, exist_ok=True)

videos = [
    r"C:\fake photos\fake video 3.mp4",
    r"C:\fake photos\fake video 4.mp4",
]

for path in videos:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    dur = n_frames / fps if fps else float("nan")
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"resolution={w}x{h} fps={fps:.2f} frames={n_frames} duration={dur:.2f}s codec={codec} size={size_mb:.2f}MB")

    # sample 6 evenly spaced frames
    idxs = np.linspace(0, max(n_frames - 1, 0), 6).astype(int)
    base = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    for i, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {idx}: FAILED TO READ")
            continue
        out_path = os.path.join(out_dir, f"{base}_f{i}_idx{idx}.png")
        cv2.imwrite(out_path, frame)
        print(f"  saved frame {idx} -> {out_path}")
    cap.release()
