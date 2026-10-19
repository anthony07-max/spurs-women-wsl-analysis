"""
Feature engineering for the xG model.

StatsBomb pitch coordinates: 120 x 80 yards, attacking left-to-right, so the
goal being attacked is at x = 120 with posts at y = 36 and y = 44.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

GOAL_X, POST_L, POST_R, GOAL_Y = 120.0, 36.0, 44.0, 40.0


def _name(d):
    """StatsBomb nests categorical values as {'id':..,'name':..}."""
    return d["name"] if isinstance(d, dict) else None


def _in_triangle(p, a, b, c):
    """True if point p lies inside triangle abc (barycentric sign test)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def freeze_frame_features(loc, frame):
    """Context from the 360-style freeze frame StatsBomb attaches to each shot:
    defenders between the ball and goal, pressure around the shooter and the
    goalkeeper's position. Returns NaNs when no freeze frame is available."""
    out = {"n_def_in_cone": np.nan, "n_opp_within_3": np.nan,
           "gk_dist_to_goal": np.nan, "gk_in_cone": np.nan, "has_freeze_frame": 0}
    if not frame:
        return out
    out["has_freeze_frame"] = 1
    post_l, post_r = (GOAL_X, POST_L), (GOAL_X, POST_R)
    n_cone = n_close = 0
    gk = None
    for p in frame:
        if p.get("teammate"):
            continue
        pl = p["location"]
        if _name(p.get("position")) == "Goalkeeper":
            gk = pl
            continue
        if _in_triangle(pl, loc, post_l, post_r):
            n_cone += 1
        if np.hypot(pl[0] - loc[0], pl[1] - loc[1]) <= 3.0:
            n_close += 1
    out["n_def_in_cone"], out["n_opp_within_3"] = n_cone, n_close
    if gk is not None:
        out["gk_dist_to_goal"] = np.hypot(GOAL_X - gk[0], GOAL_Y - gk[1])
        out["gk_in_cone"] = int(_in_triangle(gk, loc, post_l, post_r))
    else:  # keeper absent from frame (e.g. out of his goal / off-camera)
        out["gk_dist_to_goal"], out["gk_in_cone"] = 30.0, 0
    return out


def shot_row(rec):
    e, s, kp = rec["shot"], rec["shot"]["shot"], rec.get("key_pass")
    x, y = e["location"][:2]
    dx, dy = GOAL_X - x, GOAL_Y - y
    # Angle subtended by the two posts as seen from the shot location (radians).
    a1 = np.arctan2(POST_L - y, GOAL_X - x)
    a2 = np.arctan2(POST_R - y, GOAL_X - x)
    angle = abs(a2 - a1)
    kpp = (kp or {}).get("pass", {})
    if kp is None:
        assist = "Unassisted"
    elif kpp.get("cross"):
        assist = "Cross"
    elif kpp.get("cut_back"):
        assist = "Cut-back"
    elif _name(kpp.get("technique")) == "Through Ball" or kpp.get("through_ball"):
        assist = "Through ball"
    elif _name(kpp.get("type")) in ("Corner", "Free Kick"):
        assist = "Set-piece pass"
    elif _name(kpp.get("height")) == "High Pass":
        assist = "High pass"
    else:
        assist = "Ground/low pass"
    row = {
        "shot_id": e["id"], "match_id": rec["match_id"],
        "competition": rec["competition"], "season_id": rec["season_id"],
        "competition_id": rec["competition_id"], "match_date": rec["match_date"],
        "team": _name(e["team"]), "player": _name(e["player"]),
        "player_id": e["player"]["id"], "minute": e["minute"], "period": e["period"],
        "x": x, "y": y,
        "distance": float(np.hypot(dx, dy)), "angle": float(angle),
        "body_part": _name(s.get("body_part")),
        "shot_type": _name(s.get("type")),
        "technique": _name(s.get("technique")),
        "play_pattern": _name(e.get("play_pattern")),
        "under_pressure": int(bool(e.get("under_pressure"))),
        "first_time": int(bool(s.get("first_time"))),
        "one_on_one": int(bool(s.get("one_on_one"))),
        "open_goal": int(bool(s.get("open_goal"))),
        "follows_dribble": int(bool(s.get("follows_dribble"))),
        "aerial_won": int(bool(s.get("aerial_won"))),
        "assist_type": assist,
        "outcome": _name(s.get("outcome")),
        "is_goal": int(_name(s.get("outcome")) == "Goal"),
        "statsbomb_xg": s.get("statsbomb_xg"),
        "end_x": (s.get("end_location") or [np.nan])[0],
        "end_y": (s.get("end_location") or [np.nan, np.nan])[1],
    }
    row.update(freeze_frame_features(e["location"], s.get("freeze_frame")))
    return row


def load_shots(shots_dir):
    """Read every data/shots/*.jsonl file into one tidy shot-level DataFrame."""
    rows = []
    for path in sorted(glob.glob(os.path.join(shots_dir, "*.jsonl"))):
        with open(path) as fh:
            rows.extend(shot_row(json.loads(line)) for line in fh)
    df = pd.DataFrame(rows).drop_duplicates("shot_id").reset_index(drop=True)
    # Swap full legal names for the short names used in StatsBomb lineups (nicknames)
    names_path = os.path.join(shots_dir, "..", "player_names.json")
    df["player_full"] = df["player"]
    if os.path.exists(names_path):
        with open(names_path) as fh:
            nick = {int(k): v for k, v in json.load(fh).items()}
        df["player"] = df["player_id"].map(nick).fillna(df["player_full"])
    df["body_group"] = np.where(df["body_part"] == "Head", "Head",
                                np.where(df["body_part"].isin(["Left Foot", "Right Foot"]),
                                         "Foot", "Other"))
    df["set_piece"] = df["play_pattern"].isin(["From Corner", "From Free Kick",
                                               "From Throw In"]).astype(int)
    df["international"] = df["competition"].str.contains(
        "World Cup|Euro|Copa|African", regex=True).astype(int)
    return df
