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

    people = sorted({(k[1], k[2]) for k in streams if k[1] != "BALL"})
    slot_of = {person: i for i, person in enumerate(people)}
    slots = np.full((total, len(people), 3), np.nan)
    team = np.array([0 if roles.get(t) == "home" else 1 for t, _ in people], dtype=np.int8)
    for (section, team_id, person), row in streams.items():
        if team_id == "BALL":
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
    ball[hide] = np.nan
    return dict(arrays, ball=ball)


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
            labels, _ = pipeline.label(arrays, events=[], blocks=blocks)
            print(match, {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in score(labels, answers).items()}, flush=True)


if __name__ == "__main__":
    main()
