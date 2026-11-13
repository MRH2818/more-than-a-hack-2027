"""Experiment harness on the IDSSE matches (other games, real answers). Not used by run.py.

Screening is 2-fold cross-validation INSIDE the four tune matches, so the three held-out test
matches stay untouched until a finalist is confirmed (`confirm`).

Each result line: spellF1 against 3 s answers / 1 s answers / 3 s answers with 3x ball noise,
their mean, and macroF1 and possession ratio for the first.
"""

import numpy as np

from . import config, external, learned, pipeline

FOLDS = ((("J03WMX", "J03WOH"), ("J03WPY", "J03WR9")),
         (("J03WPY", "J03WR9"), ("J03WMX", "J03WOH")))
CONDITIONS = ((0.25, 3.0), (0.25, 1.0), (0.75, 3.0))      # (ball noise, answers' control rule)
_DATA = {}


def data(matches, noise, control, seed=1):
    key = (tuple(matches), noise, control, seed)
    if key not in _DATA:
        _DATA[key] = external.prepared(matches, noise, seed=seed, control_min_s=control)
    return _DATA[key]


def run_labels(arrays, blocks, model, mode="learned", post=None):
    options = dict(external.NO_VIDEO)
    if model is not None:
        options[mode] = True
    else:
        options["learned"] = False
    labels, parts = pipeline.label(arrays, events=[], blocks=blocks, options=options, model=model)
    if post is not None:
        labels = post(labels, arrays, blocks, model, parts)
    return labels


def evaluate(models_for, matches, settings=None, mode="learned", post=None):
    """models_for(match) -> model (None = hand-made v6 evidence). Returns the result row."""
    settings = settings or {}
    saved = {key: getattr(config, key) for key in settings}
    out = []
    try:
        for key, value in settings.items():
            setattr(config, key, value)
        for noise, control in CONDITIONS:
            rows = []
            for match, (arrays, answers, blocks) in data(matches, noise, control).items():
                labels = run_labels(arrays, blocks, models_for(match), mode, post)
                rows.append(external.score(labels, answers))
            out.append({"spellF1": float(np.mean([r["spellF1"] for r in rows])),
                        "macroF1": float(np.mean([r["macroF1"] for r in rows])),
                        "ratio": float(np.mean([r["n_pred"] / float(r["n_gt"]) for r in rows]))})
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
    return out


def show(name, rows):
    scores = [r["spellF1"] for r in rows]
    print("{:<34} {:.3f} / {:.3f} / {:.3f}   mean {:.3f}   macroF1 {:.3f}  ratio {:.2f}".format(
        name, scores[0], scores[1], scores[2], float(np.mean(scores)), rows[0]["macroF1"],
        rows[0]["ratio"]), flush=True)
    return float(np.mean(scores))


def screen(name, train_kwargs=None, settings=None, mode="learned", post=None, hand=False,
           models=None):
    """2-fold result inside the tune matches. train_kwargs go to learned.train."""
    kwargs = dict(control_min_s=1.0)
    kwargs.update(train_kwargs or {})
    fold_rows = []
    fitted = models if models is not None else {}
    for train_matches, test_matches in FOLDS:
        if hand:
            model = None
        else:
            if train_matches not in fitted:
                fitted[train_matches] = learned.train(train_matches, **kwargs)
            model = fitted[train_matches]
        fold_rows.append(evaluate(lambda match: model, test_matches, settings, mode, post))
    rows = [{k: float(np.mean([f[i][k] for f in fold_rows])) for k in fold_rows[0][0]}
            for i in range(len(CONDITIONS))]
    return show(name, rows), fitted


def confirm(name, train_kwargs=None, settings=None, mode="learned", post=None, hand=False):
    """Train on the four tune matches, report the three held-out test matches."""
    kwargs = dict(control_min_s=1.0)
    kwargs.update(train_kwargs or {})
    model = None if hand else learned.train(external.TUNE_MATCHES, **kwargs)
    return show("TEST " + name, evaluate(lambda match: model, external.TEST_MATCHES, settings,
                                         mode, post))
