"""Two checks on the OTHER games (IDSSE / Metrica, public data). Not used by run.py.

    python -m possession.stoppagelab definition   # which answer definition looks like the truth?
    python -m possession.stoppagelab bench        # frozen vs realistic stoppage ball, held-out
    python -m possession.stoppagelab islands      # short live stretches between two stoppages
    python -m possession.stoppagelab versions     # v6 / v7 / v9 / v10 table of TECHNICAL.md

definition: EVALUATION.md publishes what a perfect submission scores when it is 3 / 8 / 25
frames late on the full match (0.9818 / 0.8933 / 0.6539). That table depends only on how long
the true possessions are, so the same table computed on the IDSSE answers under each control
rule shows which rule gives possessions of similar length. No BAR-ATH data or labels involved.

bench: the IDSSE ball is frozen during every stoppage (config.STOPPAGE_BALL). Trains the
submitted recipe (dead-vs-live model + level-2 team model with Metrica) on the tune matches
with the frozen and with the simulated stoppage ball, and scores both on the held-out matches
under each kind of stoppage ball.
"""

import argparse
import os
import sys

import numpy as np

from . import config, external, learned, pipeline
from .ball import runs_of

sys.path.insert(0, os.path.join(config.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

PUBLISHED_SHIFT = {3: 0.9818, 8: 0.8933, 25: 0.6539}        # EVALUATION.md, full match


def _answers(raw, control):
    return raw if control == 0 else external.to_challenge_definition(raw, control)


def _late(answers, blocks, frames):
    out = answers.copy()
    for _, first, last in blocks:
        part = answers[first:last + 1]
        out[first:last + 1] = np.concatenate([np.repeat(part[:1], frames), part[:-frames]])
    return out


def definition(controls=(0, 0.5, 1.0, 2.0, 3.0)):
    print("published, full match: " + "  ".join(
        "{} frames late {:.4f}".format(k, v) for k, v in PUBLISHED_SHIFT.items()))
    loaded = [external.load(match) for match in external.match_ids()]
    for control in controls:
        rows = []
        for _, raw, blocks in loaded:
            answers = _answers(raw, control)
            codes = np.array([external.LETTER_CODES[a] for a in answers], dtype=np.int8)
            lengths = np.array([b - a for a, b, _ in
                                pm.possession_spells(codes, pm.valid_runs(codes))])
            row = {"per_match": len(lengths), "median_s": np.median(lengths) / config.FPS,
                   "under_1s": np.mean(lengths < config.FPS),
                   "under_3s": np.mean(lengths < 3 * config.FPS)}
            for frames in PUBLISHED_SHIFT:
                row[frames] = external.score(_late(answers, blocks, frames), answers)["spellF1"]
            rows.append(row)
        mean = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        print("{:<10} ".format("raw" if control == 0 else "{:g} s rule".format(control))
              + "  ".join("{} frames late {:.4f}".format(k, mean[k]) for k in PUBLISHED_SHIFT)
              + "   possessions/match {:.0f}, median {:.1f} s, under 1 s {:.0%}, under 3 s "
                "{:.0%}".format(mean["per_match"], mean["median_s"], mean["under_1s"],
                                mean["under_3s"]), flush=True)


def _with(settings, function):
    saved = {key: getattr(config, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        return function()
    finally:
        for key, value in saved.items():
            setattr(config, key, value)


def train_recipe(matches, mode):
    """The submitted recipe trained on `matches` with stoppage ball `mode`."""
    from . import metrica

    def fit():
        dead = learned.train(matches, control_min_s=config.LEARNED_CONTROL_MIN_S)
        team = learned.train(tuple(matches) + tuple(metrica.MATCHES),
                             control_min_s=config.LEARNED_CONTROL_MIN_S,
                             level=config.V8_TEAM_SETTINGS["LEARNED_LEVEL"],
                             noise=config.V8_TEAM_SETTINGS["LEARNED_NOISES"],
                             seeds=config.V8_TEAM_SETTINGS["LEARNED_SEEDS"])
        return learned.Hybrid(dead, team)
    return _with({"STOPPAGE_BALL": mode}, fit)


BENCHES = (                      # name, stoppage ball, (full, mostly) hiding rates, noise, seed
    ("frozen", "frozen", None, 0.25, 1),
    ("mix", "mix", None, 0.25, 1),
    ("place", "place", None, 0.25, 1),
    ("carry", "carry", None, 0.25, 1),
    ("mix, ball seen more", "mix", (0.25, 0.45), 0.25, 1),
    ("mix, 3x noise", "mix", None, 0.75, 1),
    ("frozen, 3x noise", "frozen", None, 0.75, 1),
    ("mix, seed 7", "mix", None, 0.25, 7),
)
CONTROLS = (0, 1.0, 3.0)


def score_bench(model, loaded, data, settings=None):
    """Mean spellF1 per answer definition plus dead-ball figures over the matches in `data`."""
    def run():
        scores = {control: [] for control in CONTROLS}
        count = {"hit": 0, "true": 0, "said": 0, "seen": 0, "seen_hit": 0, "inside": 0,
                 "stoppages": 0, "true_stoppages": 0}
        for match, arrays in data.items():
            raw, blocks = loaded[match][1], loaded[match][2]
            labels, _ = pipeline.label(arrays, events=[], blocks=blocks, model=model,
                                       options=dict(external.NO_VIDEO, learned=True))
            for control in CONTROLS:
                scores[control].append(external.score(labels, _answers(raw, control)))
            graded = raw != "X"
            truly_dead, said_dead = raw == "D", (labels == "D") & graded
            seen = ~np.isnan(arrays["ball"][:, 0])
            count["hit"] += int((truly_dead & said_dead).sum())
            count["true"] += int(truly_dead.sum())
            count["said"] += int(said_dead.sum())
            count["seen"] += int((truly_dead & seen).sum())
            count["seen_hit"] += int((truly_dead & seen & said_dead).sum())
            count["stoppages"] += len(runs_of(said_dead))
            count["true_stoppages"] += len(runs_of(truly_dead))
            for letter in ("H", "A"):
                count["inside"] += sum(1 for a, b in runs_of((labels == letter) & graded)
                                       if truly_dead[a:b].mean() > 0.5)
        return scores, count
    return _with(settings or {}, run)


def bench():
    loaded = {match: external.load(match) for match in external.TEST_MATCHES}
    data = {}
    for name, mode, rates, noise, seed in BENCHES:
        settings = {"STOPPAGE_BALL": mode}
        if rates:
            settings.update(EXT_DEAD_FULL=rates[0], EXT_DEAD_MOSTLY=rates[1])
        data[name] = _with(settings, lambda: {match: external.degrade(*loaded[match], noise, seed)
                                              for match in external.TEST_MATCHES})
    print("held-out matches; spellF1 vs raw / 1 s / 3 s answers; dead-ball frame recall "
          "(all, ball seen), precision; stoppages found / true; possessions placed inside "
          "true dead time", flush=True)
    for trained_on in ("frozen", "mix"):
        model = train_recipe(external.TUNE_MATCHES, trained_on)
        for snap in (config.SNAP_DEFAULT, 0):
            for name in data:
                scores, c = score_bench(model, loaded, data[name], {"SNAP_WINDOW": snap})
                print("trained {:<6} snap {:<3} on {:<20} {}   D recall {:.3f} (seen {:.3f}) "
                      "precision {:.3f}   stoppages {}/{}   inside dead {}".format(
                          trained_on, snap, name,
                          " / ".join("{:.3f}".format(np.mean([r["spellF1"] for r in scores[k]]))
                                     for k in CONTROLS),
                          c["hit"] / float(c["true"]), c["seen_hit"] / float(max(c["seen"], 1)),
                          c["hit"] / float(max(c["said"], 1)), c["stoppages"],
                          c["true_stoppages"], c["inside"]), flush=True)


def islands():
    """Live stretches with dead ball on both sides (probe.live_islands), per match: in the true
    answers of all seven matches, and as labelled by the recipe trained with each stoppage ball
    on the held-out matches (frozen and mix benches)."""
    from . import probe

    def per_match(label_sets):
        totals = np.zeros(len(probe.ISLAND_BINS_S))
        for labels, blocks in label_sets:
            for _, first, last in blocks:
                totals += probe.live_islands(labels[first:last + 1])
        return "  ".join("{}-{} s: {:.1f}".format(low, high, value) for (low, high), value in
                         zip(probe.ISLAND_BINS_S, totals / float(len(label_sets))))

    everything = [external.load(match) for match in external.match_ids()]
    print("per match. true answers, 7 matches:      " + per_match(
        [(answers, blocks) for _, answers, blocks in everything]), flush=True)
    loaded = {match: external.load(match) for match in external.TEST_MATCHES}
    print("           true answers, held-out 3:     " + per_match(
        [(loaded[m][1], loaded[m][2]) for m in external.TEST_MATCHES]), flush=True)
    for trained_on in ("frozen", "mix"):
        model = train_recipe(external.TUNE_MATCHES, trained_on)
        for mode in ("frozen", "mix", "carry"):
            data = _with({"STOPPAGE_BALL": mode},
                         lambda: {m: external.degrade(*loaded[m], config.EXT_BALL_NOISE_M, 1)
                                  for m in external.TEST_MATCHES})
            sets = []
            for match, arrays in data.items():
                labels, _ = pipeline.label(arrays, events=[], blocks=loaded[match][2],
                                           model=model,
                                           options=dict(external.NO_VIDEO, learned=True))
                sets.append((labels, loaded[match][2]))
            print("           trained {:<6} on {:<6} bench:  ".format(trained_on, mode)
                  + per_match(sets), flush=True)


def versions():
    """The comparison table of TECHNICAL.md: v6 / v7 / v9 / v10 on the held-out Bundesliga
    matches (models trained on the tune matches) and on the Metrica sample matches (models
    trained on all seven Bundesliga matches, no Metrica match seen)."""
    from . import metrica
    snap = config.SNAP_DEFAULT
    team_kwargs = dict(level=config.V8_TEAM_SETTINGS["LEARNED_LEVEL"],
                       noise=config.V8_TEAM_SETTINGS["LEARNED_NOISES"],
                       seeds=config.V8_TEAM_SETTINGS["LEARNED_SEEDS"])

    def single(matches, mode, **kwargs):
        return _with({"STOPPAGE_BALL": mode}, lambda: learned.train(
            matches, control_min_s=config.LEARNED_CONTROL_MIN_S, **kwargs))

    def degraded(loaded, mode, noise=config.EXT_BALL_NOISE_M):
        return _with({"STOPPAGE_BALL": mode},
                     lambda: {m: external.degrade(*loaded[m], noise, 1) for m in loaded})

    def row(name, model, loaded, data, settings, controls):
        def run():
            scores = {control: [] for control in controls}
            for match, arrays in data.items():
                labels, _ = pipeline.label(
                    arrays, events=[], blocks=loaded[match][2], model=model,
                    options=dict(external.NO_VIDEO, learned=model is not None))
                for control in controls:
                    scores[control].append(external.score(labels,
                                                          _answers(loaded[match][1], control)))
            return scores
        scores = _with(settings, run)
        print("{:<46} ".format(name) + "  ".join(
            "{}: {:.3f}".format("raw" if c == 0 else "{:g} s".format(c),
                                np.mean([r["spellF1"] for r in scores[c]])) for c in controls)
            + "   macroF1 ({:g} s) {:.3f}".format(
                controls[-1], np.mean([r["macroF1"] for r in scores[controls[-1]]])), flush=True)

    hand = {"MIN_POSSESSION": 100, "SNAP_WINDOW": snap}
    tune, everything = external.TUNE_MATCHES, tuple(external.match_ids())
    loaded = {match: external.load(match) for match in external.TEST_MATCHES}
    benches = {"mix": degraded(loaded, "mix"), "mix, 3x noise": degraded(loaded, "mix", 0.75),
               "frozen": degraded(loaded, "frozen")}
    v7 = single(tune, "frozen")
    v9 = train_recipe(tune, "frozen")
    v10 = train_recipe(tune, "mix")
    print("Bundesliga held-out matches, spellF1 by answer definition")
    for bench_name, data in benches.items():
        row("v6  on " + bench_name, None, loaded, data, hand, (0, 3.0, 1.0))
        row("v7  on " + bench_name, v7, loaded, data, {"SNAP_WINDOW": snap}, (0, 3.0, 1.0))
        row("v9  on " + bench_name, v9, loaded, data, {"SNAP_WINDOW": snap}, (0, 3.0, 1.0))
        row("v10 on " + bench_name, v10, loaded, data, {"SNAP_WINDOW": 0}, (0, 3.0, 1.0))

    loaded = {match: external.load(match) for match in metrica.MATCHES}
    benches = {"as recorded": degraded(loaded, "frozen"), "mix": degraded(loaded, "mix")}
    old_dead, new_dead = single(everything, "frozen"), single(everything, "mix")
    v9 = learned.Hybrid(old_dead, single(everything, "frozen", **team_kwargs))
    v10 = learned.Hybrid(new_dead, single(everything, "mix", **team_kwargs))
    print("Metrica sample matches, spellF1 vs raw event answers and the 1 s rule")
    for bench_name, data in benches.items():
        row("v6  on " + bench_name, None, loaded, data, hand, (0, 1.0))
        row("v7  on " + bench_name, old_dead, loaded, data, {"SNAP_WINDOW": snap}, (0, 1.0))
        row("v9  (team model without Metrica) on " + bench_name, v9, loaded, data,
            {"SNAP_WINDOW": snap}, (0, 1.0))
        row("v10 (team model without Metrica) on " + bench_name, v10, loaded, data,
            {"SNAP_WINDOW": 0}, (0, 1.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("definition", "bench", "islands", "versions"))
    args = parser.parse_args()
    {"definition": definition, "bench": bench, "islands": islands,
     "versions": versions}[args.command]()
