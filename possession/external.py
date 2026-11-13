"""Measure the pipeline on OTHER games with real answers: the open IDSSE Bundesliga data.

    python -m possession.external convert            # parse the downloaded XML -> cache npz
    python -m possession.external score              # spellF1 / macroF1 per match, as-is

Data: Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event
data in elite soccer", Scientific Data, doi:10.1038/s41597-025-04505-y, CC-BY 4.0. Seven
2022/23 German matches, TRACAB tracking at 25 fps. Downloaded from the Hugging Face mirror
(pysport/idsse-data) into cache/external/idsse/ -- local only, never committed.

Answers come from the provider's own per-frame ball fields: BallStatus 0 -> D, otherwise
BallPossession 1 -> home (H), 2 -> guest (A). These are about other games, which RULES.md allows.

Their tracking is cleaner than ours (ball almost always visible, small position error), so
`degrade` makes it look like the BAR-ATH ATD before scoring: ball position noise and ball gaps,
both set from BAR-ATH aggregates (see config, EXT_*).
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import config

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

IDSSE_DIR = os.path.join(config.CACHE_DIR, "external", "idsse")
LETTER_CODES = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


# Pinned copy of the dataset: Hugging Face revision and the first 16 hex digits of each file's
# sha256. Only match information and positions are needed (answers come from the ball frames).
HF_REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
HF_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}/{}"
FILES = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": "a884728647a564cf",
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": "5490286436c86e6f",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": "f23b3ee13b988b1f",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": "4c497fac683eee82",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": "566dd5e4c86e124a",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "e976aa0c5dba862c",
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": "429b584e89df3e3e",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": "d3dd92e509421a64",
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": "56ef948961a18e15",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": "ec4dcf30200aad69",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": "a228ef0e676d009c",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": "845202c29ad93d69",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": "cbefcb0bd7f39f3b",
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": "70b6306918901906",
}


def _sha_prefix(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def download(folder=IDSSE_DIR):
    """Fetch any missing IDSSE file (about 2.4 GB in all) and check every file's hash."""
    import shutil
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    for name, expected in FILES.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            partial = path + ".part"
            with urllib.request.urlopen(HF_URL.format(HF_REVISION, name)) as response, \
                    open(partial, "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            os.replace(partial, path)
        if _sha_prefix(path) != expected:
            raise IOError("{} does not match the pinned IDSSE file; delete it and rerun".format(
                path))


def match_ids(folder=IDSSE_DIR):
    paths = glob.glob(os.path.join(folder, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(p.rsplit("DFL-MAT-", 1)[1][:-4] for p in paths)


def _file(kind, match, folder=IDSSE_DIR):
    hits = glob.glob(os.path.join(folder, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match)))
    if not hits:
        raise IOError("missing {} file for {}".format(kind, match))
    return hits[0]


def parse(match, folder=IDSSE_DIR):
    """Arrays in the same layout as io_atd (slots normalised 0-1 + speed m/s, ball, team,
    pitch), plus answers (T,) letters and the two half blocks."""
    info = ET.parse(_file("02_01_matchinformation", match, folder)).getroot()
    roles = {team.get("TeamId"): team.get("Role") for team in info.iter("Team")}
    env = next(info.iter("Environment"))
    length, width = float(env.get("PitchX")), float(env.get("PitchY"))

    streams = {}          # (section, object) -> lists
    current = None
    for event, elem in ET.iterparse(_file("04_03_positions_raw_observed", match, folder),
                                    events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            current = (elem.get("GameSection"), elem.get("TeamId"), elem.get("PersonId"))
            streams[current] = {"N": [], "X": [], "Y": [], "S": [], "P": [], "B": []}
        elif event == "end" and elem.tag == "Frame":
            row = streams[current]
            row["N"].append(int(elem.get("N")))
            row["X"].append(float(elem.get("X")))
            row["Y"].append(float(elem.get("Y")))
            row["S"].append(float(elem.get("S", "nan")))
            if current[1] == "BALL":
                row["P"].append(int(elem.get("BallPossession", "0")))
                row["B"].append(int(elem.get("BallStatus", "0")))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            elem.clear()

    sections = [s for s in ("firstHalf", "secondHalf")
                if any(k[0] == s and k[1] == "BALL" for k in streams)]
    offsets, blocks, ball_rows = {}, [], []
    total = 0
    for half, section in enumerate(sections, 1):
        ball = streams[next(k for k in streams if k[0] == section and k[1] == "BALL")]
        n_first, n_last = ball["N"][0], ball["N"][-1]
        offsets[section] = total - n_first
        blocks.append((half, total, total + n_last - n_first))
        ball_rows.append((section, ball))
        total += n_last - n_first + 1

    # Players only: the files also carry the referees (TeamId "referee"), who must not count
    # as either team.
    people = sorted({(k[1], k[2]) for k in streams if roles.get(k[1]) in ("home", "guest")})
    slot_of = {person: i for i, person in enumerate(people)}
    slots = np.full((total, len(people), 3), np.nan)
    team = np.array([0 if roles.get(t) == "home" else 1 for t, _ in people], dtype=np.int8)
    for (section, team_id, person), row in streams.items():
        if (team_id, person) not in slot_of:
            continue
        idx = np.asarray(row["N"]) + offsets[section]
        s = slot_of[(team_id, person)]
        slots[idx, s, 0] = (np.asarray(row["X"]) + length / 2) / length
        slots[idx, s, 1] = (np.asarray(row["Y"]) + width / 2) / width
        slots[idx, s, 2] = np.asarray(row["S"]) / 3.6          # km/h -> m/s

    ball = np.full((total, 3), np.nan)
    answers = np.full(total, "X", dtype="<U1")
    for section, row in ball_rows:
        idx = np.asarray(row["N"]) + offsets[section]
        ball[idx, 0] = (np.asarray(row["X"]) + length / 2) / length
        ball[idx, 1] = (np.asarray(row["Y"]) + width / 2) / width
        status, owner = np.asarray(row["B"]), np.asarray(row["P"])
        answers[idx] = np.where(status == 0, "D", np.where(owner == 1, "H",
                                                           np.where(owner == 2, "A", "X")))
    arrays = {"slots": slots, "team": team, "ball": ball, "pitch": np.array([length, width]),
              "present": np.ones(total, dtype=bool)}
    return arrays, answers, blocks


def to_challenge_definition(answers, min_seconds=None):
    """Answers with brief uncontrolled opposing spells merged into the previous team's spell."""
    from .ball import runs_of
    min_frames = int(round((config.EXT_CONTROL_MIN_S if min_seconds is None else min_seconds)
                           * config.FPS))
    out = answers.copy()
    # Repeat until stable: merging one spell can join its neighbours into a longer one.
    while True:
        changed = False
        spells = []
        for letter in ("H", "A"):
            spells += [(a, b, letter) for a, b in runs_of(out == letter)]
        spells.sort()
        for a, b, letter in spells:
            if b - a >= min_frames or a == 0:
                continue
            before = out[a - 1]
            if before in ("H", "A") and before != letter:
                out[a:b] = before
                changed = True
        if not changed:
            return out


def cache_path(match):
    return os.path.join(config.CACHE_DIR, "external", "idsse_{}.npz".format(match))


def load(match):
    path = cache_path(match)
    if not os.path.exists(path):
        arrays, answers, blocks = parse(match)
        np.savez_compressed(path, answers=answers, blocks=np.array(blocks), **arrays)
    saved = np.load(path)
    arrays = {k: saved[k] for k in ("slots", "team", "ball", "pitch", "present")}
    return arrays, saved["answers"], [tuple(int(v) for v in b) for b in saved["blocks"]]


def sanity(arrays, answers):
    """Checks that the conversion means what we think (aggregates)."""
    from . import baseline_check
    nearest = baseline_check.nearest_player_codes(arrays)
    alive = np.isin(answers, ("H", "A")) & (nearest >= 0)
    agree = np.mean((nearest[alive] == 0) == (answers[alive] == "H")) if alive.any() else np.nan
    speed = arrays["slots"][:, :, 2]
    return {"frames": len(answers),
            "share_pct": {k: round(100.0 * float(np.mean(answers == k)), 1) for k in "HADX"},
            "nearest_player_agrees_with_owner_pct": round(100.0 * float(agree), 1),
            "player_speed_ms_p50_p95": [round(float(q), 2) for q in
                                        np.nanpercentile(speed, [50, 95])],
            "ball_seen_pct": round(100.0 * float(np.mean(~np.isnan(arrays["ball"][:, 0]))), 1)}


def correlated_noise(n, sigma, tau, rng):
    """(n, 2) AR(1) noise with stationary std `sigma` and correlation time `tau` frames."""
    a = np.exp(-1.0 / tau)
    shocks = rng.randn(n, 2) * sigma * np.sqrt(1 - a * a)
    out = np.empty((n, 2))
    out[0] = rng.randn(2) * sigma
    for t in range(1, n):
        out[t] = a * out[t - 1] + shocks[t]
    return out


def degrade(arrays, answers, blocks, noise_m, seed=0):
    """Copy of arrays whose ball looks like BAR-ATH's ATD: correlated position noise, most
    stoppages without a ball, and occasional multi-second gaps in live play."""
    from .ball import runs_of
    rng = np.random.RandomState(seed)
    pitch = arrays["pitch"]
    ball = arrays["ball"].copy()
    n = len(ball)
    if noise_m > 0:
        ball[:, :2] += correlated_noise(n, noise_m, config.EXT_NOISE_TAU, rng) / pitch
    hide = np.zeros(n, dtype=bool)
    for start, stop in runs_of(answers == "D"):
        u = rng.rand()
        if u < config.EXT_DEAD_FULL:
            hide[start:stop] = True
        elif u < config.EXT_DEAD_MOSTLY:
            length = int(round(rng.uniform(0.5, 0.99) * (stop - start)))
            if rng.rand() < 0.5:
                hide[start:start + length] = True
            else:
                hide[stop - length:stop] = True
    live = np.isin(answers, ("H", "A"))
    for _, first, last in blocks:
        frames = np.flatnonzero(live[first:last + 1]) + first
        minutes = len(frames) / float(config.FPS) / 60.0
        for _ in range(rng.poisson(config.EXT_LIVE_GAPS_PER_MIN * minutes)):
            start = int(rng.choice(frames))
            seconds = config.EXT_LIVE_GAP_MEDIAN_S * np.exp(config.EXT_LIVE_GAP_SIGMA * rng.randn())
            hide[start:min(last + 1, start + int(seconds * config.FPS))] = True
    if config.EXT_LINE_HIDE_P > 0 or config.EXT_GOAL_GAPS_PER_MIN > 0:
        length_m, width_m = (float(v) for v in pitch)
        x, y = arrays["ball"][:, 0] * length_m, arrays["ball"][:, 1] * width_m
        with np.errstate(invalid="ignore"):
            line = np.minimum(np.minimum(x, length_m - x), np.minimum(y, width_m - y))
            goal = np.minimum(x, length_m - x)
            near = line < config.EXT_LINE_HIDE_M
        for start, stop in runs_of(near):
            if rng.rand() < config.EXT_LINE_HIDE_P:
                hide[start:stop] = True
        for _, first, last in blocks:
            frames = np.flatnonzero(live[first:last + 1] & np.isfinite(goal[first:last + 1])) + first
            if not len(frames):
                continue
            weight = np.exp(-np.maximum(goal[frames], 0.0) / config.EXT_GOAL_GAP_SCALE_M)
            minutes = len(frames) / float(config.FPS) / 60.0
            count = rng.poisson(config.EXT_GOAL_GAPS_PER_MIN * minutes)
            starts = rng.choice(frames, size=count, p=weight / weight.sum()) if count else []
            for start in starts:
                seconds = config.EXT_LIVE_GAP_MEDIAN_S * np.exp(
                    config.EXT_LIVE_GAP_SIGMA * rng.randn())
                hide[start:min(last + 1, start + int(seconds * config.FPS))] = True
    ball[hide] = np.nan
    out = dict(arrays, ball=ball)
    if config.EXT_DEGRADE_PLAYERS:
        out["slots"] = degrade_players(arrays["slots"], hide | np.isnan(arrays["ball"][:, 0]),
                                       rng)
    return out


def degrade_players(slots, ball_missing, rng):
    """Copy of slots where each player drops out for a few seconds at a time, more often while
    the ball is missing, so the share of missing players matches BAR-ATH (config EXT_PLAYER_*)."""
    out = slots.copy()
    n, n_slots = slots.shape[:2]
    mean_len = config.EXT_PLAYER_DROP_MEDIAN_S * config.FPS * \
        np.exp(config.EXT_PLAYER_DROP_SIGMA ** 2 / 2.0)
    rate = np.where(ball_missing, config.EXT_PLAYER_MISSING_GAP,
                    config.EXT_PLAYER_MISSING_SEEN) / mean_len      # dropout starts per frame
    for s in range(n_slots):
        starts = np.flatnonzero(rng.rand(n) < rate)
        lengths = config.EXT_PLAYER_DROP_MEDIAN_S * config.FPS * \
            np.exp(config.EXT_PLAYER_DROP_SIGMA * rng.randn(len(starts)))
        drop = np.zeros(n + 1, dtype=np.int32)
        np.add.at(drop, starts, 1)
        np.add.at(drop, np.minimum(n, starts + np.maximum(1, lengths.astype(int))), -1)
        out[np.cumsum(drop[:n]) > 0, s] = np.nan
    return out


def ball_stats(arrays):
    """The aggregates the degradation is calibrated to (same definitions as learned.shift)."""
    from . import ball as ball_mod
    pitch = tuple(float(v) for v in arrays["pitch"])
    track = ball_mod.track(arrays["ball"], pitch)
    known = track["known"]
    xy = track["xy"]
    goal = np.minimum(xy[:, 0], pitch[0] - xy[:, 0])
    line = track["line_distance"]
    return {"missing": float(np.mean(~known)),
            "line_p10": float(np.percentile(line[known], 10)),
            "goal_p10": float(np.percentile(goal[known], 10)),
            "oob": float(np.mean(track["oob"]))}


def calibrate(matches=("J03WMX", "J03WOY"), seed=10):
    """Grid over the line-hiding probability and the goal-area gap rate; prints the four
    aggregates against BAR-ATH's (missing 0.41, line p10 2.35, goal p10 9.6, oob 0.016)."""
    saved = (config.EXT_LINE_HIDE_P, config.EXT_GOAL_GAPS_PER_MIN)
    loaded = [load(match) for match in matches]
    print("target: missing 0.410  line_p10 2.35  goal_p10 9.60  oob 0.016")
    try:
        for hide_p in (0.0, 0.5, 0.7, 0.8, 0.9, 1.0):
            for rate in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
                config.EXT_LINE_HIDE_P, config.EXT_GOAL_GAPS_PER_MIN = hide_p, rate
                rows = [ball_stats(degrade(a, ans, b, config.EXT_BALL_NOISE_M, seed))
                        for a, ans, b in loaded]
                mean = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
                print("hide_p {:.1f} goal_gaps/min {:.1f}:  missing {:.3f}  line_p10 {:5.2f}  "
                      "goal_p10 {:5.2f}  oob {:.3f}".format(hide_p, rate, mean["missing"],
                                                            mean["line_p10"], mean["goal_p10"],
                                                            mean["oob"]), flush=True)
    finally:
        config.EXT_LINE_HIDE_P, config.EXT_GOAL_GAPS_PER_MIN = saved


def near_mode(arrays):
    """BAR-ATH-comparable 'ball at a player's feet' distance mode (m), as in possession.probe."""
    from . import ball as ball_mod, evidence
    track = ball_mod.track(arrays["ball"], tuple(arrays["pitch"]))
    d_home, d_away = evidence.team_distances(arrays["slots"], arrays["team"], track["xy"],
                                             tuple(arrays["pitch"]))
    _, info = evidence.calibrate_zone(d_home, d_away, track["observed"] & ~track["oob"])
    return info.get("near_mode_m", float("nan"))


def calibrate_noise(match, grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    arrays, answers, blocks = load(match)
    modes = {s: near_mode(degrade(arrays, answers, blocks, s)) for s in grid}
    best = min(modes, key=lambda s: abs(modes[s] - config.EXT_TARGET_NEAR_MODE_M))
    return best, modes


def score(labels, answers):
    gt = np.array([LETTER_CODES[a] for a in answers], dtype=np.int8)
    pred = np.array([LETTER_CODES[a] for a in labels], dtype=np.int8)
    runs = pm.valid_runs(gt)
    pred = np.where(gt == pm.X, pm.X, pred)
    cm = pm.scored_confusion(pm.confusion(gt, pm.fill_abstentions(pred, runs)))
    spells = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return {"spellF1": spells["mean_f1"], "macroF1": pm.macro_f1(cm),
            "n_gt": spells["n_gt"], "n_pred": spells["n_pred"]}


NO_VIDEO = {"videoball": False, "videofull": False, "whistle": False}


# Matches used to choose settings, and matches held out to test the choice honestly.
TUNE_MATCHES = ("J03WMX", "J03WOH", "J03WPY", "J03WR9")
TEST_MATCHES = ("J03WN1", "J03WOY", "J03WQQ")
SEARCH = {
    "SWITCH_TURNOVER": (1.0, 2.5, 5.0, 10.0, 20.0),
    "SWITCH_TO_DEAD": (1.0, 2.0, 4.0, 8.0),
    "SWITCH_FROM_DEAD": (1.0, 2.0, 4.0, 8.0),
    "MIN_POSSESSION": (6, 12, 25, 50),
    "MIN_DEAD": (12, 25, 50, 100),
}


def prepared(matches, noise, seed=0, control_min_s=None):
    """Degraded arrays plus answers in the challenge's definition (control_min_s=0: raw DFL)."""
    out = {}
    for match in matches:
        arrays, answers, blocks = load(match)
        degraded = degrade(arrays, answers, blocks, noise, seed)
        target = answers if control_min_s == 0 else to_challenge_definition(answers, control_min_s)
        out[match] = (degraded, target, blocks)
    return out


def evaluate(settings, data, options=None):
    """Mean spellF1 / macroF1 / predicted-vs-true possession ratio with config overrides."""
    from . import pipeline
    # Other games have no video and must never read BAR-ATH's detections or homography.
    options = dict(options or {}, videoball=False, videofull=False, whistle=False)
    saved = {key: getattr(config, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        rows = []
        for arrays, answers, blocks in data.values():
            labels, _ = pipeline.label(arrays, events=[], blocks=blocks, options=options)
            rows.append(score(labels, answers))
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
    return {"spellF1": float(np.mean([r["spellF1"] for r in rows])),
            "macroF1": float(np.mean([r["macroF1"] for r in rows])),
            "poss_ratio": float(np.mean([r["n_pred"] / float(r["n_gt"]) for r in rows]))}


SEARCH_OPTIONS = ("touches", "zone", "lineaware", "cleandead", "softcontrol")
SEARCH2 = {
    "SWITCH_TURNOVER": (0.0, 0.25, 0.5, 1.0, 2.0),
    "SWITCH_TO_DEAD": (2.0, 4.0, 8.0),
    "SWITCH_FROM_DEAD": (4.0, 8.0, 16.0),
    "MIN_POSSESSION": (25, 50, 75, 100),
    "MIN_DEAD": (25, 50, 100),
}
START2 = {"SWITCH_TURNOVER": 1.0, "SWITCH_TO_DEAD": 4.0, "SWITCH_FROM_DEAD": 8.0,
          "MIN_POSSESSION": 50, "MIN_DEAD": 50}


def _fmt(result):
    return {k: round(v, 3) for k, v in result.items()}


def tune(noise, rounds=2, search=None, start=None, with_options=False, control_min_s=None):
    """Coordinate search on TUNE_MATCHES (settings, optionally also on/off components);
    report the result on TEST_MATCHES."""
    from . import pipeline
    search = search or SEARCH
    tune_data = prepared(TUNE_MATCHES, noise, control_min_s=control_min_s)
    test_data = prepared(TEST_MATCHES, noise, control_min_s=control_min_s)
    current = dict(start) if start else {key: getattr(config, key) for key in search}
    options = {key: v for key, v in pipeline.resolve_options().items()
               if key in SEARCH_OPTIONS} if with_options else {}
    best = evaluate(current, tune_data, options)
    print("start   ", current, options, _fmt(best), flush=True)
    for _ in range(rounds):
        trials = [(key, value) for key, values in search.items() for value in values]
        if with_options:
            trials += [("option:" + key, not options[key]) for key in SEARCH_OPTIONS]
        for key, value in trials:
            if key.startswith("option:"):
                trial_settings, trial_options = current, dict(options, **{key[7:]: value})
            elif value == current[key]:
                continue
            else:
                trial_settings, trial_options = dict(current, **{key: value}), options
            result = evaluate(trial_settings, tune_data, trial_options)
            if result["spellF1"] > best["spellF1"] + 1e-6:
                current, options, best = trial_settings, trial_options, result
                print("better  ", current, options, _fmt(best), flush=True)
    default = {key: getattr(config, key) for key in search}
    print("\nchosen on tune matches:", current, options)
    print("held-out matches, default:", _fmt(evaluate(default, test_data)))
    print("held-out matches, chosen: ", _fmt(evaluate(current, test_data, options)))
    return current, options


VARIANTS = {
    "current": {},
    "no-cleandead": {"cleandead": False},
    "touches": {"touches": True},
    "softcontrol": {"softcontrol": True},
    "zone": {"zone": True},
    "restarts": {"restarts": True},
    "ballfree": {"ballfree": True},
    "gapdead": {"gapdead": True},
    "lineaware": {"lineaware": True},
}


def variants(noise, settings=None):
    """Every optional component, scored on all seven matches with real answers."""
    data = prepared(match_ids(), noise)
    for name, options in VARIANTS.items():
        result = evaluate(settings or {}, data, options)
        print("{:<13} {}".format(name, {k: round(v, 3) for k, v in result.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("calibrate")
    sub.add_parser("convert")
    p_score = sub.add_parser("score")
    p_score.add_argument("--raw", action="store_true", help="score on the clean tracking")
    p_score.add_argument("--noise", type=float, default=None)
    sub.add_parser("noise")
    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--noise", type=float, default=None)
    p_tune.add_argument("--control-min", type=float, default=None,
                        help="seconds for the challenge-definition conversion (0 = raw DFL)")
    p_tune.add_argument("--joint", action="store_true",
                        help="wider settings grid plus on/off components, from the first result")
    p_var = sub.add_parser("variants")
    p_var.add_argument("--noise", type=float, default=None)
    args = parser.parse_args()
    if args.command == "variants":
        variants(config.EXT_BALL_NOISE_M if args.noise is None else args.noise)
        return
    if args.command == "tune":
        noise = config.EXT_BALL_NOISE_M if args.noise is None else args.noise
        if args.joint:
            tune(noise, search=SEARCH2, start=START2, with_options=True,
                 control_min_s=args.control_min)
        else:
            tune(noise)
        return
    if args.command == "calibrate":
        calibrate()
        return
    if args.command == "download":
        download()
        print("IDSSE files present and verified")
        return
    from . import pipeline
    if args.command == "noise":
        best, modes = calibrate_noise(match_ids()[0])
        print("near mode by noise sigma:", {k: round(v, 2) for k, v in modes.items()})
        print("chosen sigma:", best)
        return
    for match in match_ids():
        arrays, answers, blocks = load(match)
        if args.command == "convert":
            print(match, sanity(arrays, answers), flush=True)
        else:
            if not args.raw:
                noise = config.EXT_BALL_NOISE_M if args.noise is None else args.noise
                arrays = degrade(arrays, answers, blocks, noise)
            labels, _ = pipeline.label(arrays, events=[], blocks=blocks, options=NO_VIDEO)
            print(match, {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in score(labels, answers).items()}, flush=True)


if __name__ == "__main__":
    main()
