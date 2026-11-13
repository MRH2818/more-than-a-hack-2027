"""Referee whistles from the tactical video's own audio track, detected locally.

    python -m possession.audio features      # decode audio -> cache/audio_features.npz
    python -m possession.audio check         # whistle counts and how they line up (aggregates)

No training and no labels: a whistle is a strong, steady tone in the 2-5 kHz band, which broad
crowd noise is not. Nothing leaves this machine; features are cached in cache/ (never committed).
"""

import argparse
import os
import sys

import numpy as np

from . import config
from .ball import runs_of

sys.path.insert(0, config.ROOT)
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402


def decode_mono(path, rate=config.AUDIO_RATE):
    """(samples float32 mono at `rate`, audio start minus video start in seconds)."""
    import av
    container = av.open(path)
    stream = container.streams.audio[0]
    video = container.streams.video[0] if container.streams.video else None

    def start(s):
        return float(s.start_time * s.time_base) if s is not None and s.start_time is not None \
            else 0.0

    offset = start(stream) - start(video)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    chunks = []
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().reshape(-1))
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray().reshape(-1))
    container.close()
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0 if chunks else np.zeros(0)
    return audio, offset


def frame_features(audio, offset, n_frames, rate=config.AUDIO_RATE, window=config.AUDIO_WINDOW,
                   band=config.WHISTLE_BAND_HZ, fps=config.FPS, chunk=5000):
    """Per video frame: tonality (dB), peak frequency (Hz), band share (dB), loudness (dB).

    Frame f is centred at (f + 0.5) / fps seconds of video time; NaN where no audio covers it.
    """
    freqs = np.fft.rfftfreq(window, 1.0 / rate)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    hann = np.hanning(window).astype(np.float32)
    out = {name: np.full(n_frames, np.nan) for name in
           ("tonality_db", "peak_hz", "band_share_db", "level_db")}
    centres = np.round(((np.arange(n_frames) + 0.5) / fps - offset) * rate).astype(np.int64)
    starts = centres - window // 2
    ok = (starts >= 0) & (starts + window <= len(audio))
    for first in range(0, n_frames, chunk):
        idx = np.flatnonzero(ok[first:first + chunk]) + first
        if not len(idx):
            continue
        windows = audio[starts[idx][:, None] + np.arange(window)[None, :]] * hann
        power = np.abs(np.fft.rfft(windows, axis=1)) ** 2 + 1e-12
        band_power = power[:, in_band]
        peak = np.argmax(band_power, axis=1)
        out["tonality_db"][idx] = 10 * np.log10(band_power[np.arange(len(idx)), peak]
                                                / np.median(band_power, axis=1))
        out["peak_hz"][idx] = freqs[in_band][peak]
        out["band_share_db"][idx] = 10 * np.log10(band_power.sum(axis=1) / power.sum(axis=1))
        out["level_db"][idx] = 10 * np.log10(power.sum(axis=1) / window)
    return out


def robust_z(x):
    finite = x[np.isfinite(x)]
    if not len(finite):
        return np.full(len(x), np.nan)
    median = np.median(finite)
    mad = 1.4826 * np.median(np.abs(finite - median)) or 1e-9
    return (x - median) / mad


def whistles(features, z=config.WHISTLE_Z, min_len=config.WHISTLE_MIN,
             max_len=config.WHISTLE_MAX, pitch_std=config.WHISTLE_PITCH_STD_HZ):
    """[(start, stop)] half-open frame spans of whistle events."""
    with np.errstate(invalid="ignore"):
        hot = robust_z(features["tonality_db"]) >= z
    # Close single-frame holes so one blast is one event.
    holes = ~hot[1:-1] & hot[:-2] & hot[2:]
    hot[1:-1] |= holes
    events = []
    for start, stop in runs_of(hot):
        if not (min_len <= stop - start <= max_len):
            continue
        if np.nanstd(features["peak_hz"][start:stop]) > pitch_std:
            continue
        events.append((start, stop))
    return events


def load_features(path=config.AUDIO_FEATURES):
    if not os.path.exists(path):
        return None
    saved = np.load(path)
    return {key: saved[key] for key in saved.files}


def boundary_bonus(events, n, pre=config.WHISTLE_PRE, post=config.WHISTLE_POST,
                   to_dead=config.SWITCH_TO_DEAD, from_dead=config.SWITCH_FROM_DEAD):
    """(entry (n, 3), exit (n, 3)) decoder bonuses: entering D / leaving D near a whistle."""
    entry = np.zeros((n, 3))
    exit_ = np.zeros((n, 3))
    for start, _ in events:
        lo, hi = max(0, start - pre), min(n, start + post + 1)
        entry[lo:hi, 2] = to_dead
        exit_[lo:hi, 2] = from_dead
    return entry, exit_


def dead_evidence(events, n, post=config.WHISTLE_POST, weight=config.WHISTLE_DEAD_LL):
    """(n,) dead log-odds for the second after each whistle starts."""
    out = np.zeros(n)
    for start, _ in events:
        out[max(0, start):min(n, start + post)] = weight
    return out


def build_features(out_path=config.AUDIO_FEATURES):
    path = require(["video"], GAME)[0]
    audio, offset = decode_mono(path)
    n = source_frames("tactical")
    features = frame_features(audio, offset, n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **features)
    covered = np.isfinite(features["tonality_db"])
    print("audio seconds {:.2f}  offset vs video {:+.3f} s  frames covered {:.2%}".format(
        len(audio) / float(config.AUDIO_RATE), offset, covered.mean()))


def check():
    from . import io_atd, pipeline, tagging
    features = load_features()
    if features is None:
        print("no features yet: python -m possession.audio features")
        return
    n = len(features["tonality_db"])
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    events = whistles(features)
    tz = robust_z(features["tonality_db"])
    print("tonality z  p50 {:.2f}  p99 {:.2f}  p99.9 {:.2f}  frames >= threshold {:.2%}".format(
        *np.nanpercentile(tz, [50, 99, 99.9]), np.nanmean(tz >= config.WHISTLE_Z)))
    print("whistle events: {} total, {} in graded play".format(
        len(events), sum(graded[s] for s, _ in events)))
    for half, first, last in blocks:
        inside = [e for e in events if first <= e[0] <= last]
        print("  half {}: {} events, {:.2f} per minute".format(
            half, len(inside), len(inside) / ((last - first + 1) / config.FPS / 60.0)))
    if events:
        lengths = np.array([b - a for a, b in events]) / float(config.FPS)
        pitch = np.array([np.nanmedian(features["peak_hz"][a:b]) for a, b in events])
        print("  duration s p10/p50/p90 {}   pitch Hz p10/p50/p90 {}".format(
            np.round(np.percentile(lengths, [10, 50, 90]), 2).tolist(),
            np.round(np.percentile(pitch, [10, 50, 90])).tolist()))

    events_tags = tagging.load_events()
    kickoffs = [e["start_frame"] for e in events_tags
                if (e["code"] or "").strip().upper() == "SET PIECES"
                and (e["tags"].get("Type") or "").strip().lower().startswith("kick off")]
    starts = np.array([s for s, _ in events])
    near = sum(bool(len(starts)) and np.min(np.abs(starts - k)) <= 5 * config.FPS
               for k in kickoffs)
    print("kickoff tags with a whistle within 5 s: {} of {}".format(near, len(kickoffs)))

    # Is anything whistle-like there at all? Strongest tonality within 5 s of each kickoff tag,
    # against the strongest within 10 s windows at random graded positions (fixed seed).
    half_window = 5 * config.FPS

    def window_max(centre):
        values = tz[max(0, centre - half_window):centre + half_window]
        return float(np.nanmax(values)) if np.isfinite(values).any() else np.nan

    at_kickoffs = np.array([window_max(k) for k in kickoffs])
    rng = np.random.RandomState(0)
    random_max = np.array([window_max(c) for c in rng.choice(np.flatnonzero(graded), 500)])
    print("max tonality z within 5 s of each kickoff: {}".format(
        np.round(np.sort(at_kickoffs), 2).tolist()))
    print("same for 500 random 10 s windows: p50 {:.2f}  p90 {:.2f}  p99 {:.2f}".format(
        *np.nanpercentile(random_max, [50, 90, 99])))
    print("share of random windows each kickoff beats: {}".format(
        [round(float(np.mean(random_max < v)), 2) for v in np.sort(at_kickoffs)]))

    labels, _ = pipeline.label(io_atd.load(), events=events_tags)
    dead = labels == "D"
    d_starts = np.flatnonzero(dead[1:] & ~dead[:-1]) + 1
    d_stops = np.flatnonzero(~dead[1:] & dead[:-1]) + 1
    bounds = np.sort(np.concatenate([d_starts, d_stops]))
    graded_events = [s for s, _ in events if graded[s]]
    if graded_events and len(bounds):
        gaps = np.array([np.min(np.abs(bounds - s)) for s in graded_events]) / float(config.FPS)
        print("graded whistles vs current stoppage boundaries: within 2 s {:.1%}  within 5 s "
              "{:.1%}  inside a stoppage {:.1%}".format(
                  np.mean(gaps <= 2), np.mean(gaps <= 5),
                  np.mean([dead[s] for s in graded_events])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("features")
    sub.add_parser("check")
    args = parser.parse_args()
    build_features() if args.command == "features" else check()


if __name__ == "__main__":
    main()
