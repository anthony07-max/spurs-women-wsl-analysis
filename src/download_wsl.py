"""
Download StatsBomb Open Data for the FA Women's Super League (2018/19 to
2020/21) and turn each match into four compact tables:

  data/wsl/matches.csv          one row per match (season, date, teams, score)
  data/wsl/shots/37_<sid>.jsonl every Shot plus its key pass, in the same format
                                as the xG training data, so src/features.py reads both
  data/wsl/team_match.parquet   one row per team per match (chance creation counts)
  data/wsl/player_match.parquet one row per player per match (minutes, positions,
                                on-ball and defensive actions), built the same way as
                                github.com/anthony07-max/football-scouting-tool

Raw event files (~3-4 MB per match) are processed on the fly and never stored.
Data (c) StatsBomb, used under their open-data licence.

Usage:  python src/download_wsl.py
"""
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wsl")
COMP_ID = 37
SEASONS = [(4, "2018/19"), (42, "2019/20"), (90, "2020/21")]
SET_PIECES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}
GOAL = (120.0, 40.0)
session = requests.Session()


def get_json(url, retries=5):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise


def clock(s):
    m, sec = s.split(":")
    return int(m) + int(sec) / 60


def dist_goal(loc):
    return math.hypot(GOAL[0] - loc[0], GOAL[1] - loc[1])


def in_box(loc):
    return loc[0] >= 102 and 18 <= loc[1] <= 62


def progressive(start, end):
    """Ball moved at least 25% closer to the centre of the goal and at least 5 yards nearer."""
    d0, d1 = dist_goal(start), dist_goal(end)
    return d1 <= 0.75 * d0 and (d0 - d1) >= 5


def sca_qualifies(e):
    """Successful attacking actions that can create a shot (FBref-style definition,
    carries excluded): completed passes (open play or set piece), completed dribbles,
    fouls won and shots (a shot that leads to a rebound shot)."""
    t = e["type"]["name"]
    if t == "Pass":
        return "outcome" not in e["pass"]
    if t == "Dribble":
        return e["dribble"].get("outcome", {}).get("name") == "Complete"
    return t in ("Foul Won", "Shot")


DEF_KEYS = ["tackles", "interceptions", "recoveries", "pressures", "blocks", "clearances"]
COUNT_COLS = (
    ["passes", "passes_cmp", "long_passes", "long_passes_cmp", "crosses", "crosses_cmp",
     "through_balls", "prog_passes", "passes_final_third", "passes_into_box", "key_passes",
     "key_passes_sp", "assists", "xa", "sca", "carries", "prog_carries", "carries_into_box",
     "carries_final_third", "receipts", "box_receipts", "dribbles", "dribbles_cmp", "shots",
     "npxg", "np_goals", "shots_on_target", "headed_shots", "pens", "pen_goals", "aerials_won",
     "aerials_lost", "tackles_att", "pressures_att3", "miscontrols", "dispossessed", "fouls",
     "fouls_won", "dribbled_past", "def_x_sum", "def_n"]
    + DEF_KEYS + [k + "_padj" for k in DEF_KEYS]
)
TEAM_COLS = ["passes", "passes_cmp", "prog_passes", "passes_into_box", "passes_final_third",
             "key_passes", "key_passes_sp", "sca", "prog_carries", "carries_into_box",
             "box_receipts", "shots_np", "goals"]


def process_match(match, season):
    mid = match["match_id"]
    events = get_json(f"{BASE}/events/{mid}.json")
    lineups = get_json(f"{BASE}/lineups/{mid}.json")
    events.sort(key=lambda e: e["index"])
    by_id = {e["id"]: e for e in events}

    # ---- shots in xG-training format ------------------------------------------
    shot_rows = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        kp_id = e["shot"].get("key_pass_id")
        shot_rows.append({
            "match_id": mid, "shot": e, "key_pass": by_id.get(kp_id) if kp_id else None,
            "competition_id": COMP_ID, "season_id": match["season"]["season_id"],
            "competition": f"FA WSL {season}", "match_date": match["match_date"],
            "home_team": match["home_team"]["home_team_name"],
            "away_team": match["away_team"]["away_team_name"],
        })

    # ---- match length and possession proxy (share of passes) -------------------
    period_end = defaultdict(float)
    for e in events:
        period_end[e["period"]] = max(period_end[e["period"]], e["minute"] + e["second"] / 60)
    match_end = period_end[max(p for p in period_end if p <= 4)]
    team_passes = defaultdict(int)
    for e in events:
        if e["type"]["name"] == "Pass":
            team_passes[e["team"]["id"]] += 1
    tot = sum(team_passes.values()) or 1

    # ---- shot-creating actions: the two qualifying actions before each non-penalty shot
    sca_ids = defaultdict(int)
    poss_events = defaultdict(list)
    for e in events:
        poss_events[e["possession"]].append(e)
    for evs in poss_events.values():
        for i, e in enumerate(evs):
            if e["type"]["name"] != "Shot" or e["shot"]["type"]["name"] == "Penalty":
                continue
            found = 0
            for prev in reversed(evs[:i]):
                if found == 2:
                    break
                if prev["team"]["id"] == e["team"]["id"] and sca_qualifies(prev):
                    sca_ids[prev["id"]] += 1
                    found += 1

    shot_xg = {}
    for e in events:
        if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty":
            kp = e["shot"].get("key_pass_id")
            if kp:
                shot_xg[kp] = e["shot"].get("statsbomb_xg", 0.0)

    team_ids = [t["team_id"] for t in lineups]
    team_name = {t["team_id"]: t["team_name"] for t in lineups}
    team = {tid: dict(match_id=mid, season=season, match_date=match["match_date"], team_id=tid,
                      team=team_name[tid], opponent=team_name[[x for x in team_ids if x != tid][0]],
                      home=int(tid == match["home_team"]["home_team_id"]),
                      possession=team_passes[tid] / tot, **{k: 0 for k in TEAM_COLS})
            for tid in team_ids}

    rows = {}
    for t in lineups:
        tid = t["team_id"]
        poss = team_passes[tid] / tot
        for p in t["lineup"]:
            if not p["positions"]:
                continue
            off = [clock(c["time"]) for c in p.get("cards", [])
                   if c["card_type"] in ("Red Card", "Second Yellow")]
            cap = min(off) if off else match_end
            pos_min = defaultdict(float)
            for s in p["positions"]:
                a = clock(s["from"])
                b = min(clock(s["to"]) if s["to"] else match_end, cap)
                pos_min[s["position"]] += max(0.0, b - a)
            minutes = sum(pos_min.values())
            if minutes <= 0:
                continue
            rows[p["player_id"]] = dict(
                match_id=mid, season=season, match_date=match["match_date"],
                team_id=tid, team=team_name[tid], player_id=p["player_id"],
                player=p["player_name"], nickname=p.get("player_nickname"),
                minutes=minutes, team_poss=poss,
                started=p["positions"][0]["start_reason"] == "Starting XI",
                positions=";".join(f"{k}={v:.1f}" for k, v in pos_min.items()),
                **{k: 0.0 for k in COUNT_COLS},
            )

    blank = {k: 0.0 for k in COUNT_COLS}
    for e in events:
        tid = e.get("team", {}).get("id")
        if tid not in team:
            continue
        T = team[tid]
        pl = e.get("player")
        r = rows.get(pl["id"]) if pl else None
        if r is None:
            r = dict(blank, team_poss=T["possession"])   # still count team totals
        typ = e["type"]["name"]
        loc = e.get("location")
        padj = 0.5 / max(1 - r["team_poss"], 0.2)  # scale defensive actions to a 50% possession game
        n_sca = sca_ids.get(e["id"], 0)
        r["sca"] += n_sca
        T["sca"] += n_sca

        def defend(key):
            r[key] += 1
            r[key + "_padj"] += padj
            if loc:
                r["def_x_sum"] += loc[0]
                r["def_n"] += 1

        if typ == "Pass":
            pa = e["pass"]
            ptype = pa.get("type", {}).get("name")
            complete = "outcome" not in pa
            end = pa["end_location"]
            if pa.get("aerial_won"):
                r["aerials_won"] += 1
            if ptype in SET_PIECES:
                if pa.get("shot_assist") or pa.get("goal_assist"):
                    r["key_passes_sp"] += 1
                    T["key_passes_sp"] += 1
                continue
            r["passes"] += 1; T["passes"] += 1
            r["passes_cmp"] += complete; T["passes_cmp"] += complete
            if pa.get("length", 0) >= 32:
                r["long_passes"] += 1
                r["long_passes_cmp"] += complete
            if pa.get("cross"):
                r["crosses"] += 1
                r["crosses_cmp"] += complete
            if pa.get("through_ball") or pa.get("technique", {}).get("name") == "Through Ball":
                r["through_balls"] += 1
            if complete and progressive(loc, end) and loc[0] >= 30:
                r["prog_passes"] += 1; T["prog_passes"] += 1
            if complete and loc[0] < 80 <= end[0]:
                r["passes_final_third"] += 1; T["passes_final_third"] += 1
            if complete and in_box(end) and not in_box(loc):
                r["passes_into_box"] += 1; T["passes_into_box"] += 1
            if pa.get("shot_assist") or pa.get("goal_assist"):
                r["key_passes"] += 1; T["key_passes"] += 1
            if pa.get("goal_assist"):
                r["assists"] += 1
            r["xa"] += shot_xg.get(e["id"], 0.0)
        elif typ == "Carry":
            end = e["carry"]["end_location"]
            if loc and math.hypot(end[0] - loc[0], end[1] - loc[1]) >= 5:
                r["carries"] += 1
                if progressive(loc, end) and loc[0] >= 30:
                    r["prog_carries"] += 1; T["prog_carries"] += 1
                if in_box(end) and not in_box(loc):
                    r["carries_into_box"] += 1; T["carries_into_box"] += 1
                if loc[0] < 80 <= end[0]:
                    r["carries_final_third"] += 1
        elif typ == "Ball Receipt*":
            if "outcome" not in e.get("ball_receipt", {}):
                r["receipts"] += 1
                if loc and in_box(loc):
                    r["box_receipts"] += 1; T["box_receipts"] += 1
        elif typ == "Dribble":
            r["dribbles"] += 1
            if e["dribble"].get("outcome", {}).get("name") == "Complete":
                r["dribbles_cmp"] += 1
        elif typ == "Shot":
            sh = e["shot"]
            goal = sh["outcome"]["name"] == "Goal"
            if e["period"] <= 4:
                T["goals"] += goal
            if sh["type"]["name"] == "Penalty":
                r["pens"] += 1
                r["pen_goals"] += goal
                continue
            T["shots_np"] += 1
            r["shots"] += 1
            r["npxg"] += sh.get("statsbomb_xg", 0.0)
            r["np_goals"] += goal
            r["shots_on_target"] += sh["outcome"]["name"] in ("Goal", "Saved", "Saved To Post")
            if sh.get("body_part", {}).get("name") == "Head":
                r["headed_shots"] += 1
            if sh.get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Duel":
            d = e["duel"]
            dt = d.get("type", {}).get("name")
            if dt == "Aerial Lost":
                r["aerials_lost"] += 1
            elif dt == "Tackle":
                r["tackles_att"] += 1
                if d.get("outcome", {}).get("name") in ("Won", "Success", "Success In Play", "Success Out"):
                    defend("tackles")
        elif typ == "Interception":
            if e["interception"].get("outcome", {}).get("name") not in ("Lost", "Lost In Play", "Lost Out"):
                defend("interceptions")
        elif typ == "Ball Recovery":
            if not e.get("ball_recovery", {}).get("recovery_failure"):
                defend("recoveries")
        elif typ == "Pressure":
            defend("pressures")
            if loc and loc[0] >= 80:
                r["pressures_att3"] += 1
        elif typ == "Block":
            defend("blocks")
        elif typ == "Clearance":
            defend("clearances")
            if e.get("clearance", {}).get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Miscontrol":
            r["miscontrols"] += 1
            if e.get("miscontrol", {}).get("aerial_won"):
                r["aerials_won"] += 1
        elif typ == "Dispossessed":
            r["dispossessed"] += 1
        elif typ == "Foul Committed":
            r["fouls"] += 1
        elif typ == "Foul Won":
            r["fouls_won"] += 1
        elif typ == "Dribbled Past":
            r["dribbled_past"] += 1

    return shot_rows, list(team.values()), list(rows.values())


def main():
    os.makedirs(os.path.join(OUT, "shots"), exist_ok=True)
    needed = ["matches.csv", "team_match.parquet", "player_match.parquet"]
    if all(os.path.exists(os.path.join(OUT, f)) for f in needed):
        print("WSL data already downloaded; delete data/wsl/ to refresh")
        return
    match_rows, team_rows, player_rows, failed = [], [], [], []
    shots_by_season = defaultdict(list)
    for sid, season in SEASONS:
        ms = get_json(f"{BASE}/matches/{COMP_ID}/{sid}.json")
        print(f"FA WSL {season}: {len(ms)} matches in StatsBomb Open Data", flush=True)
        for m in ms:
            match_rows.append(dict(match_id=m["match_id"], season=season, match_date=m["match_date"],
                                   home_team=m["home_team"]["home_team_name"],
                                   away_team=m["away_team"]["away_team_name"],
                                   home_score=m["home_score"], away_score=m["away_score"]))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(process_match, m, season): m["match_id"] for m in ms}
            for f in as_completed(futs):
                try:
                    s, t, p = f.result()
                except Exception as exc:  # pragma: no cover
                    failed.append((futs[f], repr(exc)))
                    continue
                shots_by_season[sid] += s
                team_rows += t
                player_rows += p
    for sid, rows in shots_by_season.items():
        with open(os.path.join(OUT, "shots", f"{COMP_ID}_{sid}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    pd.DataFrame(team_rows).to_parquet(os.path.join(OUT, "team_match.parquet"), index=False)
    pd.DataFrame(player_rows).to_parquet(os.path.join(OUT, "player_match.parquet"), index=False)
    names = {r["player_id"]: r["nickname"] or r["player"] for r in player_rows}
    with open(os.path.join(OUT, "player_names.json"), "w") as fh:
        json.dump(names, fh, ensure_ascii=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(OUT, "matches.csv"), index=False)
    print(f"Saved {len(match_rows)} matches, {sum(map(len, shots_by_season.values()))} shots, "
          f"{len(player_rows):,} player-match rows; failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
