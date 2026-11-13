"""Ball detection on the tactical video, run locally. Prints aggregates only.

    python -m possession.video_ball info                 # frame count, fps, resolution
    python -m possession.video_ball info --count         # also decode every frame to count them
    python -m possession.video_ball benchmark            # speed + agreement with the ATD ball
    python -m possession.video_ball detect               # all graded frames -> cache/ (resumable)
    python -m possession.video_ball detect --start 40000 --count 500 --out cache/a.npz
    python -m possession.video_ball compare cache/a.npz cache/b.npz   # determinism check

Rules this file keeps (RULES.md, FAQ.md, CLAUDE.md):
- everything runs on this machine; no frame, crop or detection leaves it;
- detections are never edited by hand and never committed (cache/ and models/ are ignored);
- the detector is a public pre-trained model (Ultralytics YOLO11, COCO class "sports ball").

Pixel -> pitch uses the provided per-frame homography (starter/load_homography.py), which maps
the ground plane, so a ball in the air lands in the wrong place; the benchmark measures how
often detections agree with the provided tracking where both exist.
"""

import argparse
import os
import sys
import time

import numpy as np

from . import config

sys.path.insert(0, config.ROOT)
sys.path.insert(0, os.path.join(config.ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402

MODELS_DIR = os.path.join(config.ROOT, "models")
# Public weights (Ultralytics assets release v8.4.0, AGPL-3.0), pinned by hash so a fresh clone
# detects with exactly the same model.
WEIGHTS = {
    "yolo11s.pt": ("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt",
                   "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
}


def ensure_weights(name="yolo11s.pt"):
    """Download the pinned public weights if absent; refuse a file whose hash differs."""
    import hashlib
    import urllib.request
    path = os.path.join(MODELS_DIR, name)
    url, expected = WEIGHTS[name]
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print("downloading {} ...".format(url), flush=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != expected:
        raise RuntimeError("{} has sha256 {}, expected {}".format(path, digest, expected))
    return path
SPORTS_BALL = 32  # COCO class id
# A detection whose projected position is this far outside the pitch (normalised units) is
# ignored: off-pitch balls (spare balls, crowd) are not the match ball.
OFF_PITCH_MARGIN = 0.05


def video_path(game=GAME):
    return require(["video"], game)[0]


def open_video(path):
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise IOError("cannot open video: {}".format(path))
    return capture


def info(count=False):
    import cv2
    path = video_path()
    capture = open_video(path)
    out = {
        "frames_in_header": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 3),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "expected_frames": source_frames("tactical"),
    }
    if count:
        began, n = time.time(), 0
        while capture.grab():
            n += 1
        out["frames_decoded"] = n
        out["count_seconds"] = round(time.time() - began, 1)
    capture.release()
    for key, value in out.items():
        print("{:<20} {}".format(key, value))
    frames = out.get("frames_decoded", out["frames_in_header"])
    print("MATCHES submission length" if frames == out["expected_frames"]
          else "DOES NOT MATCH the submission length -- stop and investigate, do not pad or trim")


class Detector:
    """One YOLO model at one input size on one device. Returns the best ball per frame."""

    def __init__(self, weights, imgsz, device, conf):
        from ultralytics import YOLO, settings
        settings.update({"sync": False})  # no usage analytics leave this machine
        self.model = YOLO(os.path.join(MODELS_DIR, weights))
        self.imgsz, self.device, self.conf = imgsz, device, conf
        self.name = "{}@{}/{}".format(weights.replace(".pt", ""), imgsz, device)

    def __call__(self, image):
        """(pixel_x, pixel_y, confidence, n_candidates) of the most confident ball, or None."""
        result = self.model.predict(image, imgsz=self.imgsz, device=self.device,
                                    conf=self.conf, classes=[SPORTS_BALL], verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        best = int(np.argmax(conf))
        x0, y0, x1, y1 = xyxy[best]
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1), float(conf[best]), len(conf))


def to_pitch(matrix, pixel):
    """Pixel -> normalised pitch (0-1) through the inverse homography, or None if off-pitch."""
    from load_homography import unproject
    if pixel is None or np.isnan(matrix).any():
        return None
    x, y = unproject(matrix, pixel[0], pixel[1])
    x, y = float(x), float(y)
    if not (-OFF_PITCH_MARGIN <= x <= 1 + OFF_PITCH_MARGIN
            and -OFF_PITCH_MARGIN <= y <= 1 + OFF_PITCH_MARGIN):
        return None
    return x, y


def sample_chunks(observed, graded, n_chunks, length, seed):
    """Start frames of n_chunks windows inside graded play: half starting where the ATD ball is
    seen, half where it is missing (so both cases are measured)."""
    rng = np.random.RandomState(seed)
    ok = graded.copy()
    ok[-length:] = False
    starts = []
    for want_seen in [True, False] * (n_chunks // 2) + [True] * (n_chunks % 2):
        pool = np.flatnonzero(ok & (observed == want_seen))
        if not len(pool):
            continue
        start = int(rng.choice(pool))
        starts.append(start)
        ok[max(0, start - length):start + length] = False
    return sorted(starts)


def benchmark(configs, n_chunks, length, seed, max_shift):
    import cv2
    from load_homography import load as load_homography
    from . import io_atd

    arrays = io_atd.load()
    pitch = np.asarray(arrays["pitch"], dtype=np.float64)
    atd_ball = arrays["ball"][:, :2]
    observed = ~np.isnan(atd_ball[:, 0])
    n = len(observed)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    matrices, _, valid = load_homography(source="tactical")

    detectors = [Detector(w, s, d, c) for w, s, d, c in configs]
    starts = sample_chunks(observed, graded, n_chunks, length, seed)
    capture = open_video(video_path())
    # found[name][frame] = (x, y) normalised pitch, for frames with an on-pitch ball
    found = {d.name: {} for d in detectors}
    timing = {d.name: [] for d in detectors}
    multi = {d.name: 0 for d in detectors}
    decode_seconds, frames_seen, seek_misses = 0.0, [], 0

    for start in starts:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
            seek_misses += 1
        for frame in range(start, start + length):
            began = time.time()
            ok, image = capture.read()
            decode_seconds += time.time() - began
            if not ok:
                break
            frames_seen.append(frame)
            for detector in detectors:
                began = time.time()
                hit = detector(image)
                timing[detector.name].append(time.time() - began)
                if hit is not None and hit[3] > 1:
                    multi[detector.name] += 1
                if hit is not None and valid[frame]:
                    xy = to_pitch(matrices[frame], hit)
                    if xy is not None:
                        found[detector.name][frame] = xy
        print("chunk at {:.0%} of the video done".format(start / float(n)), flush=True)
    capture.release()

    frames_seen = np.array(frames_seen)
    seen = frames_seen[observed[frames_seen]]
    missing = frames_seen[~observed[frames_seen]]
    print("\nframes tested: {} ({} with ATD ball, {} without) in {} chunks; "
          "decode {:.3f} s/frame; seeks not landing exactly: {}".format(
              len(frames_seen), len(seen), len(missing), len(starts),
              decode_seconds / max(1, len(frames_seen)), seek_misses))

    graded_missing = int((graded & ~observed).sum())
    for detector in detectors:
        name = detector.name
        hits = found[name]
        per_frame = float(np.median(timing[name])) if timing[name] else float("nan")

        def errors(shift):
            out = []
            for frame in seen:
                if frame in hits and 0 <= frame + shift < n and observed[frame + shift]:
                    d = (np.array(hits[frame]) - atd_ball[frame + shift]) * pitch
                    out.append(float(np.hypot(d[0], d[1])))
            return np.array(out)

        shifts = {s: errors(s) for s in range(-max_shift, max_shift + 1)}
        best_shift = min(shifts, key=lambda s: np.median(shifts[s]) if len(shifts[s]) else 1e9)
        err = shifts[0]
        found_seen = np.mean([f in hits for f in seen]) if len(seen) else float("nan")
        found_missing = np.mean([f in hits for f in missing]) if len(missing) else float("nan")
        print("\n== {} ==".format(name))
        print("  median seconds per frame        {:.3f}  -> full graded-missing run ~{:.1f} h"
              .format(per_frame, per_frame * graded_missing / 3600.0))
        print("  ball found where ATD has one    {:.1%}".format(found_seen))
        if len(err):
            print("  distance to ATD ball (m)        p50 {:.2f}  p90 {:.2f}  within 2 m {:.1%}  "
                  "within 5 m {:.1%}".format(np.median(err), np.percentile(err, 90),
                                             np.mean(err <= 2), np.mean(err <= 5)))
            print("  best frame shift vs ATD         {:+d} (median error there {:.2f} m)".format(
                best_shift, float(np.median(shifts[best_shift]))))
        print("  ball found where ATD has none   {:.1%}".format(found_missing))
        print("  frames with >1 ball candidate   {}".format(multi[name]))


# The setup chosen by the benchmark: small model, native resolution, Apple GPU (same aggregates
# as on CPU, ~5x faster). Determinism across runs is checked with `compare`.
DETECT_CONFIG = ("yolo11s.pt", 1920, "mps", 0.10)
TOP_K = 5
CHECKPOINT_EVERY = 2000
BATCH = 8
# Ultralytics gives up on non-max suppression after ~2.4 s for a batch of 8 and returns partial
# boxes -- which only happens when the machine stalls (e.g. waking from sleep) and would make the
# output depend on timing. A batch slower than this is therefore simply run again.
STALL_SECONDS = 2.0
MAX_RETRIES = 5
DEFAULT_OUT = os.path.join(config.CACHE_DIR, "video_ball_yolo11s_1920.npz")


def detect_all(image_batch, model_config=DETECT_CONFIG, top_k=TOP_K, _cache={}):
    """(len(batch), top_k, 5) candidates per image: pixel x, pixel y, conf, width, height;
    NaN-padded, sorted by confidence."""
    if "model" not in _cache:
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
        _cache["model"] = YOLO(os.path.join(MODELS_DIR, model_config[0]))
    weights, imgsz, device, conf = model_config
    results = _cache["model"].predict(image_batch, imgsz=imgsz, device=device, conf=conf,
                                      classes=[SPORTS_BALL], verbose=False)
    out = np.full((len(image_batch), top_k, 5), np.nan, dtype=np.float32)
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        order = np.argsort(-confs, kind="stable")[:top_k]
        for j, b in enumerate(order):
            x0, y0, x1, y1 = xyxy[b]
            out[i, j] = (0.5 * (x0 + x1), 0.5 * (y0 + y1), confs[b], x1 - x0, y1 - y0)
    return out


def detect_stable(images):
    """detect_all, rerun whenever the batch took long enough that NMS may have been cut short."""
    for _ in range(MAX_RETRIES):
        began = time.time()
        out = detect_all(images)
        if time.time() - began < STALL_SECONDS:
            return out
        print("slow batch (machine stalled?) -- running it again", flush=True)
    raise RuntimeError("detector kept stalling; is the machine overloaded or asleep?")


def detect(out_path, start=None, count=None, redo=False):
    """Run the detector over graded frames (or [start, start+count)), saving candidates to
    out_path every CHECKPOINT_EVERY frames; rerunning resumes where it stopped. With redo, frames
    in the range are detected again even if already done."""
    n = source_frames("tactical")
    wanted = np.zeros(n, dtype=bool)
    if start is not None:
        wanted[start:start + (count or 1)] = True
    else:
        for _, first, last in half_video_blocks(GAME):
            wanted[first:last + 1] = True

    candidates = np.full((n, TOP_K, 5), np.nan, dtype=np.float32)
    done = np.zeros(n, dtype=bool)
    if os.path.exists(out_path):
        saved = np.load(out_path)
        candidates, done = saved["candidates"], saved["done"]
        if redo:
            done = done.copy()
            done[wanted] = False
            print("redoing {} frames".format(int(wanted.sum())))
        else:
            print("resuming: {} frames already done".format(int((done & wanted).sum())))
    todo = np.flatnonzero(wanted & ~done)
    if not len(todo):
        print("nothing to do")
        return

    def save():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, candidates=candidates, done=done,
                            config=np.array([str(c) for c in DETECT_CONFIG]))
        os.replace(tmp, out_path)

    import cv2
    capture = open_video(video_path())
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(todo[0]))
    position = int(todo[0])
    began, processed, since_save = time.time(), 0, 0
    batch_frames, batch_images = [], []
    last = int(todo[-1])
    while position <= last:
        if not wanted[position] or done[position]:
            if not capture.grab():
                break
            position += 1
            continue
        ok, image = capture.read()
        if not ok:
            break
        batch_frames.append(position)
        batch_images.append(image)
        position += 1
        if len(batch_images) == BATCH or position > last:
            candidates[batch_frames] = detect_stable(batch_images)
            done[batch_frames] = True
            processed += len(batch_frames)
            since_save += len(batch_frames)
            batch_frames, batch_images = [], []
            if since_save >= CHECKPOINT_EVERY:
                save()
                since_save = 0
                rate = processed / (time.time() - began)
                left = int((wanted & ~done).sum())
                print("{:>7} frames done, {:.1f} frames/s, ~{:.0f} min left".format(
                    int((done & wanted).sum()), rate, left / rate / 60.0), flush=True)
    if batch_images:
        candidates[batch_frames] = detect_stable(batch_images)
        done[batch_frames] = True
    capture.release()
    save()
    with_ball = ~np.isnan(candidates[:, 0, 0]) & done & wanted
    print("finished: {} frames, ball candidate on {:.1%}, {:.1f} min".format(
        int((done & wanted).sum()), with_ball.sum() / max(1, (done & wanted).sum()),
        (time.time() - began) / 60.0))


def compare(path_a, path_b):
    """Determinism check: are two detection runs identical on the frames both cover?"""
    a, b = np.load(path_a), np.load(path_b)
    both = a["done"] & b["done"]
    ca, cb = a["candidates"][both], b["candidates"][both]
    same = np.array_equal(np.isnan(ca), np.isnan(cb)) and np.array_equal(
        np.nan_to_num(ca), np.nan_to_num(cb))
    diff = np.nanmax(np.abs(ca - cb)) if np.isfinite(ca - cb).any() else 0.0
    print("frames compared: {}".format(int(both.sum())))
    print("identical: {}   max abs difference: {:.6g}".format(same, float(diff)))


DEFAULT_CONFIGS = [
    ("yolo11s.pt", 1280, "cpu", 0.10),
    ("yolo11s.pt", 1920, "cpu", 0.10),
    ("yolo11m.pt", 1920, "cpu", 0.10),
    ("yolo11s.pt", 1920, "mps", 0.10),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_info = sub.add_parser("info")
    p_info.add_argument("--count", action="store_true")
    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--chunks", type=int, default=8)
    p_bench.add_argument("--length", type=int, default=25)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--max-shift", type=int, default=5)
    p_detect = sub.add_parser("detect")
    p_detect.add_argument("--out", default=DEFAULT_OUT)
    p_detect.add_argument("--start", type=int, default=None)
    p_detect.add_argument("--count", type=int, default=None)
    p_detect.add_argument("--redo", action="store_true",
                          help="detect frames in the range again even if already done")
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("a")
    p_compare.add_argument("b")
    args = parser.parse_args()
    if args.command == "info":
        info(args.count)
    elif args.command == "benchmark":
        benchmark(DEFAULT_CONFIGS, args.chunks, args.length, args.seed, args.max_shift)
    elif args.command == "detect":
        detect(args.out, args.start, args.count, args.redo)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
