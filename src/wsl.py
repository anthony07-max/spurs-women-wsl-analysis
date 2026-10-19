"""
Helpers for the Spurs Women WSL analysis: loading the WSL tables built by
download_wsl.py, team-season summaries with league ranks, and per-90 player
profiles with position-group percentiles (the same approach as
github.com/anthony07-max/football-scouting-tool).
"""
import os
import re

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wsl")
SPURS = "Spurs"

POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "CM", "Left Defensive Midfield": "CM", "Right Defensive Midfield": "CM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "AM/W", "Left Attacking Midfield": "AM/W", "Right Attacking Midfield": "AM/W",
    "Left Wing": "AM/W", "Right Wing": "AM/W", "Left Midfield": "AM/W", "Right Midfield": "AM/W",
    "Center Forward": "ST", "Left Center Forward": "ST", "Right Center Forward": "ST", "Secondary Striker": "ST",
}
GROUP_LABEL = {"CB": "centre-backs", "FB": "full-backs and wing-backs", "CM": "central and defensive midfielders",
               "AM/W": "attacking midfielders and wingers", "ST": "strikers"}


def short_team(name):
    """StatsBomb uses several spellings for the same club ('Manchester United' and
    'Manchester United W'); strip the suffixes so each club has one name."""
    if "Tottenham" in name:
        return SPURS
    name = re.sub(r"\s+(WFC|LFC|FCW|W|Women|Ladies)$", "", name.strip())
    return (name.replace("Brighton & Hove Albion", "Brighton").replace("Manchester", "Man")
                .replace("West Ham United", "West Ham"))


def load_tables():
    matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
    for c in ("home_team", "away_team"):
        matches[c] = matches[c].map(short_team)
    tm = pd.read_parquet(os.path.join(DATA, "team_match.parquet"))
    tm["team"] = tm.team.map(short_team)
    tm["opponent"] = tm.opponent.map(short_team)
    # goals from the official score (the event count misses own goals)
    sc = matches.set_index("match_id")
    home = tm.home == 1
    tm["goals"] = np.where(home, tm.match_id.map(sc.home_score), tm.match_id.map(sc.away_score))
    pm = pd.read_parquet(os.path.join(DATA, "player_match.parquet"))
    pm["team"] = pm.team.map(short_team)
    return matches, tm, pm


def team_season_table(team_match):
    """Per-match averages for every team-season, for and against, with ranks
    within each season (1 = best; for 'against' metrics lower is better)."""
    tm = team_match.copy()
    opp = tm[["match_id", "team", "npxg", "sb_npxg", "shots_np", "goals", "np_goals",
              "key_passes", "sca", "prog_passes", "box_receipts"]]
    opp = opp.rename(columns={c: c + "_against" for c in opp.columns if c not in ("match_id",)})
    tm = tm.merge(opp, left_on=["match_id", "opponent"], right_on=["match_id", "team_against"])
    tm["points"] = np.select([tm.goals > tm.goals_against, tm.goals == tm.goals_against], [3, 1], 0)
    agg = tm.groupby(["season", "team"]).agg(
        matches=("match_id", "nunique"), points=("points", "sum"),
        goals=("goals", "sum"), goals_against=("goals_against", "sum"),
        np_goals=("np_goals", "sum"), np_goals_against=("np_goals_against", "sum"),
        npxg=("npxg", "sum"), npxg_against=("npxg_against", "sum"),
        sb_npxg=("sb_npxg", "sum"), sb_npxg_against=("sb_npxg_against", "sum"),
        shots=("shots_np", "sum"), shots_against=("shots_np_against", "sum"),
        key_passes=("key_passes", "sum"), key_passes_against=("key_passes_against", "sum"),
        sca=("sca", "sum"), sca_against=("sca_against", "sum"),
        prog_passes=("prog_passes", "sum"), prog_passes_against=("prog_passes_against", "sum"),
        box_receipts=("box_receipts", "sum"), possession=("possession", "mean"),
    ).reset_index()
    out = agg[["season", "team", "matches", "points", "goals", "goals_against", "np_goals",
               "np_goals_against", "possession"]].copy()
    per_match = ["npxg", "npxg_against", "sb_npxg", "sb_npxg_against", "shots", "shots_against",
                 "key_passes", "key_passes_against", "sca", "sca_against", "prog_passes",
                 "prog_passes_against", "box_receipts"]
    for c in per_match:
        out[c + "_pm"] = agg[c] / agg.matches
    out["points_pm"] = agg.points / agg.matches
    out["npxgd_pm"] = out.npxg_pm - out.npxg_against_pm
    out["xg_per_shot"] = agg.npxg / agg.shots
    out["xg_per_shot_against"] = agg.npxg_against / agg.shots_against
    out["np_goals_minus_xg"] = agg.np_goals - agg.npxg
    out["np_goals_against_minus_xg"] = agg.np_goals_against - agg.npxg_against
    lower_better = {c for c in out.columns if "against" in c}
    rank_cols = [c for c in out.columns if c.endswith("_pm") or c.startswith("xg_per_shot")] + ["possession"]
    for c in rank_cols:
        out[c + "_rank"] = out.groupby("season")[c].rank(ascending=c in lower_better, method="min").astype(int)
    return out


# ---- player profiles ---------------------------------------------------------
FEATURES = {
    "npxg_90": "Non-penalty xG",
    "shots_90": "Shots",
    "xg_per_shot": "xG per shot",
    "box_receipts_90": "Touches in box",
    "xa_90": "Open-play xA",
    "key_passes_90": "Key passes",
    "sca_90": "Shot-creating actions",
    "passes_into_box_90": "Passes into box",
    "crosses_90": "Crosses",
    "prog_passes_90": "Progressive passes",
    "prog_carries_90": "Progressive carries",
    "dribbles_90": "Dribbles attempted",
    "dribble_pct": "Dribble success %",
    "passes_90": "Passes attempted",
    "pass_pct": "Pass completion %",
    "long_passes_90": "Long passes",
    "long_pass_pct": "Long pass completion %",
    "turnovers_90": "Miscontrols + dispossessed",
    "tackles_padj_90": "Tackles won (PAdj)",
    "interceptions_padj_90": "Interceptions (PAdj)",
    "pressures_padj_90": "Pressures (PAdj)",
    "pressures_att3_90": "Final-third pressures",
    "recoveries_padj_90": "Ball recoveries (PAdj)",
    "blocks_padj_90": "Blocks (PAdj)",
    "clearances_padj_90": "Clearances (PAdj)",
    "aerials_won_90": "Aerial duels won",
    "aerial_pct": "Aerial win %",
}
LOWER_IS_BETTER = {"turnovers_90"}
RADAR = {
    "CB": ["passes_90", "pass_pct", "long_passes_90", "long_pass_pct", "prog_passes_90", "prog_carries_90",
           "tackles_padj_90", "interceptions_padj_90", "blocks_padj_90", "clearances_padj_90",
           "pressures_padj_90", "aerials_won_90", "aerial_pct"],
    "FB": ["crosses_90", "xa_90", "sca_90", "passes_into_box_90", "prog_passes_90", "prog_carries_90",
           "dribbles_90", "pass_pct", "tackles_padj_90", "interceptions_padj_90", "pressures_padj_90",
           "recoveries_padj_90", "aerial_pct"],
    "CM": ["passes_90", "pass_pct", "prog_passes_90", "prog_carries_90", "key_passes_90", "sca_90", "npxg_90",
           "dribbles_90", "turnovers_90", "tackles_padj_90", "interceptions_padj_90", "pressures_padj_90",
           "recoveries_padj_90"],
    "AM/W": ["npxg_90", "shots_90", "xa_90", "key_passes_90", "sca_90", "passes_into_box_90", "crosses_90",
             "prog_passes_90", "prog_carries_90", "dribbles_90", "dribble_pct", "turnovers_90",
             "pressures_att3_90"],
    "ST": ["npxg_90", "shots_90", "xg_per_shot", "box_receipts_90", "xa_90", "key_passes_90", "sca_90",
           "dribbles_90", "dribble_pct", "prog_carries_90", "turnovers_90", "pressures_att3_90", "aerial_pct"],
}


def _shrunk_rate(success, attempts, prior_rate, k):
    """Shrink a percentage towards the population rate with k pseudo-attempts."""
    return 100 * (success + k * prior_rate) / (attempts + k)


def build_profiles(pm, min_minutes=900):
    """One per-90 profile per player per season (outfield players with at least
    `min_minutes` WSL minutes in the StatsBomb data for that season)."""
    pos = pm[["player_id", "season", "positions"]].assign(p=pm.positions.str.split(";")).explode("p")
    pos[["position", "mins"]] = pos.p.str.split("=", expand=True)
    pos["mins"] = pos.mins.astype(float)
    pos["group"] = pos.position.map(POSITION_GROUP)
    gmins = pos.pivot_table(index=["player_id", "season"], columns="group", values="mins",
                            aggfunc="sum", fill_value=0)
    keys = ["player_id", "season"]
    count_cols = [c for c in pm.columns if pm[c].dtype.kind in "fi"
                  and c not in ("match_id", "team_id", "player_id", "minutes", "team_poss")]
    agg = pm.groupby(keys)[count_cols + ["minutes"]].sum()
    agg["matches"] = pm.groupby(keys).match_id.nunique()
    main = (pm.groupby(keys + ["team"]).minutes.sum().reset_index()
              .sort_values("minutes").groupby(keys).tail(1).set_index(keys))
    agg["team"] = main.team
    names = pm.groupby(keys).agg(player=("player", "first"), nickname=("nickname", "first"))
    agg = agg.join(names)
    agg["name"] = agg.nickname.fillna(agg.player)
    agg["group"] = gmins.drop(columns="GK", errors="ignore").idxmax(axis=1)
    agg.loc[gmins.idxmax(axis=1) == "GK", "group"] = "GK"
    agg = agg[(agg.minutes >= min_minutes) & (agg.group != "GK")].copy()

    p90 = agg.minutes / 90
    for c in ["npxg", "shots", "box_receipts", "xa", "key_passes", "sca", "passes_into_box", "crosses",
              "prog_passes", "prog_carries", "dribbles", "passes", "long_passes", "pressures_att3",
              "aerials_won", "np_goals"] + [k + "_padj" for k in
                                            ["tackles", "interceptions", "pressures", "recoveries",
                                             "blocks", "clearances"]]:
        agg[c + "_90"] = agg[c] / p90
    agg["turnovers_90"] = (agg.miscontrols + agg.dispossessed) / p90
    league = agg.sum(numeric_only=True)
    agg["pass_pct"] = _shrunk_rate(agg.passes_cmp, agg.passes, league.passes_cmp / league.passes, 10)
    agg["long_pass_pct"] = _shrunk_rate(agg.long_passes_cmp, agg.long_passes,
                                        league.long_passes_cmp / league.long_passes, 10)
    agg["dribble_pct"] = _shrunk_rate(agg.dribbles_cmp, agg.dribbles, league.dribbles_cmp / league.dribbles, 10)
    aer = agg.aerials_won + agg.aerials_lost
    agg["aerial_pct"] = _shrunk_rate(agg.aerials_won, aer, league.aerials_won /
                                     (league.aerials_won + league.aerials_lost), 10)
    agg["xg_per_shot"] = (agg.npxg + 3 * 0.10) / (agg.shots + 3)
    return agg.reset_index()


def percentiles(profiles, idx):
    """Percentile (0-100) of one player-season against every WSL player-season in
    the same position group; inverted where lower is better."""
    row = profiles.loc[idx]
    peers = profiles[profiles.group == row.group]
    out = {}
    for c in RADAR[row.group]:
        pct = (peers[c] < row[c]).mean() * 100 + (peers[c] == row[c]).mean() * 50
        out[c] = 100 - pct if c in LOWER_IS_BETTER else pct
    return out
