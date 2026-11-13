"""Metrica Sports' public sample matches as extra games with answers (other games, allowed).

Source: https://github.com/metrica-sports/sample-data (three anonymised matches, tracking +
events, "if you use it for anything public, please acknowledge the source"). Same provider as
this challenge's data, so its event semantics are the closest public reference for what
"possession" and "ball out of play" mean here.

    python -m possession.metrica            # download if needed, convert, print a sanity summary

Answers are derived by code, never by hand:
  D        the tracking has no ball (in these files the ball is NaN while play is stopped)
  H / A    the team of the latest on-ball event (PASS, SHOT, RECOVERY, SET PIECE, BALL LOST);
           a lost ball stays with the team that lost it until the other team's RECOVERY, as in
           CHALLENGE.md ("possession stays with the last team that had it")
Match ids: MET1, MET2, MET3. Arrays have the same layout as io_atd / external.parse.
"""

import json
import os
import sys

import numpy as np

from . import config

FOLDER = os.path.join(config.CACHE_DIR, "external", "metrica")
URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/Sample_Game_{}/{}"
REVISION = "master"
PITCH = (105.0, 68.0)
ON_BALL = ("PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")
FILES = {
    1: ("Sample_Game_1_RawTrackingData_Home_Team.csv", "Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1_RawEventsData.csv"),
    2: ("Sample_Game_2_RawTrackingData_Home_Team.csv", "Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2_RawEventsData.csv"),
    3: ("Sample_Game_3_tracking.txt", "Sample_Game_3_metadata.xml", "Sample_Game_3_events.json"),
}
MATCHES = ("MET1", "MET2", "MET3")


def download(folder=FOLDER):
    import shutil
    import urllib.request
    os.makedirs(folder, exist_ok=True)
    for game, names in FILES.items():
        for name in names:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                continue
            print("downloading", name, flush=True)
            with urllib.request.urlopen(URL.format(REVISION, game, name)) as response, \
                    open(path + ".part", "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            os.replace(path + ".part", path)


def _speeds(xy_norm):
    """(T, S) speed in m/s from normalised positions: central differences, 5-frame mean."""
    x = xy_norm[:, :, 0] * PITCH[0]
    y = xy_norm[:, :, 1] * PITCH[1]
    speed = np.full(x.shape, np.nan)
    speed[1:-1] = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) * config.FPS / 2.0
    kernel = np.ones(5) / 5.0
    out = np.full(x.shape, np.nan)
    for s in range(x.shape[1]):
        v = speed[:, s]
        ok = np.isfinite(v)
        if ok.sum() < 5:
            continue
        num = np.convolve(np.where(ok, v, 0.0), kernel, mode="same")
        den = np.convolve(ok.astype(float), kernel, mode="same")
        out[:, s] = np.where(ok & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return np.minimum(out, 12.0)


def _answers(n, ball_seen, events, periods):
    """events: [(start_frame_index, team 0/1)] sorted; periods: [(first, last)] frame indices."""
    owner = np.full(n, -1, dtype=np.int8)
    for first, last in periods:
        inside = [(f, t) for f, t in events if first <= f <= last]
        if not inside:
            continue
        owner[first:last + 1] = inside[0][1]
        for f, t in inside:
            owner[f:last + 1] = t
    answers = np.full(n, "X", dtype="<U1")
    answers[owner == 0] = "H"
    answers[owner == 1] = "A"
    answers[(owner >= 0) & ~ball_seen] = "D"
    return answers


def _parse_csv(game, folder):
    import pandas as pd
    home_name, away_name, events_name = FILES[game]
    home = pd.read_csv(os.path.join(folder, home_name), skiprows=2)
    away = pd.read_csv(os.path.join(folder, away_name), skiprows=2)
    n = len(home)
    h = home.iloc[:, 3:-2].to_numpy(dtype=np.float64).reshape(n, -1, 2)
    a = away.iloc[:, 3:-2].to_numpy(dtype=np.float64).reshape(n, -1, 2)
    xy = np.concatenate([h, a], axis=1)
    team = np.array([0] * h.shape[1] + [1] * a.shape[1], dtype=np.int8)
    ball = home.iloc[:, -2:].to_numpy(dtype=np.float64)
    period = home["Period"].to_numpy()
    periods = [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]))
               for p in sorted(set(period.tolist()))]
    ev = pd.read_csv(os.path.join(folder, events_name))
    ev = ev[ev["Type"].isin(ON_BALL)]
    events = sorted((int(f) - 1, 0 if t == "Home" else 1)
                    for f, t in zip(ev["Start Frame"], ev["Team"]))
    return xy, team, ball, periods, events


def _parse_epts(folder):
    """Game 3 (EPTS/FIFA format), read with kloppy (the reader Metrica recommends)."""
    from kloppy import metrica as kloppy_metrica
    track_name, meta_name, events_name = FILES[3]
    dataset = kloppy_metrica.load_tracking_epts(meta_data=os.path.join(folder, meta_name),
                                                raw_data=os.path.join(folder, track_name),
                                                coordinates="metrica")
    players, team = [], []
    for side, kloppy_team in enumerate(dataset.metadata.teams):
        for player in kloppy_team.players:
            players.append(player)
            team.append(side)
    slot_of = {player.player_id: i for i, player in enumerate(players)}
    team_ids = [kloppy_team.team_id for kloppy_team in dataset.metadata.teams]
    records = dataset.records
    first_id = records[0].frame_id
    n = records[-1].frame_id - first_id + 1
    xy = np.full((n, len(players), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    period_frames = {}
    for record in records:
        t = record.frame_id - first_id
        period_frames.setdefault(record.period.id, []).append(t)
        if record.ball_coordinates is not None:
            ball[t] = (record.ball_coordinates.x, record.ball_coordinates.y)
        for player, data in record.players_data.items():
            if data.coordinates is not None:
                xy[t, slot_of[player.player_id]] = (data.coordinates.x, data.coordinates.y)
    periods = [(min(v), max(v)) for _, v in sorted(period_frames.items())]
    with open(os.path.join(folder, events_name)) as handle:
        raw = json.load(handle)["data"]
    events = sorted((e["start"]["frame"] - first_id, team_ids.index(e["team"]["id"]))
                    for e in raw if e["type"]["name"] in ON_BALL and e.get("team"))
    return xy, np.array(team, dtype=np.int8), ball, periods, events


def cache_path(match):
    return os.path.join(config.CACHE_DIR, "external", "metrica_{}.npz".format(match))


def load(match, folder=FOLDER):
    """(arrays, answers, blocks) for MET1 / MET2 / MET3, cached."""
    path = cache_path(match)
    if not os.path.exists(path):
        download(folder)
        game = int(match[3:])
        xy, team, ball, periods, events = _parse_epts(folder) if game == 3 \
            else _parse_csv(game, folder)
        n = len(xy)
        slots = np.concatenate([xy, _speeds(xy)[:, :, None]], axis=2)
        ball3 = np.concatenate([ball, np.full((n, 1), np.nan)], axis=1)
        answers = _answers(n, ~np.isnan(ball[:, 0]), events, periods)
        blocks = np.array([(i + 1, a, b) for i, (a, b) in enumerate(periods)])
        np.savez_compressed(path, slots=slots, team=team, ball=ball3, pitch=np.array(PITCH),
                            present=np.ones(n, dtype=bool), answers=answers, blocks=blocks)
    saved = np.load(path)
    arrays = {k: saved[k] for k in ("slots", "team", "ball", "pitch", "present")}
    return arrays, saved["answers"], [tuple(int(v) for v in b) for b in saved["blocks"]]


def main():
    from .ball import runs_of
    for match in MATCHES:
        arrays, answers, blocks = load(match)
        live = np.isin(answers, ("H", "A"))
        spells = sum(len(runs_of(answers == k)) for k in "HA")
        minutes = sum(b - a + 1 for _, a, b in blocks) / 25.0 / 60.0
        seen = (~np.isnan(arrays["slots"][:, :, 0]))
        print(match, {"frames": len(answers), "halves": [(a, b) for _, a, b in blocks],
                      "share_pct": {k: round(100.0 * float(np.mean(answers == k)), 1) for k in "HADX"},
                      "possessions_per_min": round(spells / minutes, 2),
                      "players_per_team": [round(float(seen[:, arrays["team"] == s].sum(1)[live].mean()), 2)
                                           for s in (0, 1)],
                      "speed_ms_p50_p95": [round(float(v), 2) for v in
                                           np.nanpercentile(arrays["slots"][:, :, 2], [50, 95])]},
              flush=True)


if __name__ == "__main__":
    main()
