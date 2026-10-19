"""
Training data for the xG model: the same men's StatsBomb Open Data shots used in
github.com/anthony07-max/football-xg-model (copied from that repo's
src/download_data.py, with the output folder changed to data/train_shots/).

Keeps every Shot event plus the pass that assisted it.

Raw event files are large (~3-4 MB per match), so instead of keeping them we
extract shots on the fly and write a compact JSON-lines file per competition
season to data/shots/. Data (c) StatsBomb, used under their open-data licence.

Usage:  python src/download_training_shots.py
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train_shots")

# (competition_id, season_id, label) - men's competitions with many shots.
COMPETITIONS = [
    (2, 27, "Premier League 2015/16"),
    (11, 27, "La Liga 2015/16"),
    (9, 27, "Bundesliga 2015/16"),
    (12, 27, "Serie A 2015/16"),
    (7, 27, "Ligue 1 2015/16"),
    (43, 3, "FIFA World Cup 2018"),
    (43, 106, "FIFA World Cup 2022"),
    (55, 43, "UEFA Euro 2020"),
    (55, 282, "UEFA Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "African Cup of Nations 2023"),
]
# La Liga seasons from StatsBomb's Messi collection (2015/16 already above).
for sid, label in [(37, "2004/05"), (38, "2005/06"), (39, "2006/07"), (40, "2007/08"),
                   (41, "2008/09"), (21, "2009/10"), (22, "2010/11"), (23, "2011/12"),
                   (24, "2012/13"), (25, "2013/14"), (26, "2014/15"), (2, "2016/17"),
                   (1, "2017/18"), (4, "2018/19"), (42, "2019/20"), (90, "2020/21")]:
    COMPETITIONS.append((11, sid, f"La Liga {label}"))

session = requests.Session()


def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise


def shots_for_match(match):
    """Return compact shot records (shot event + key pass) for a single match."""
    events = get_json(f"{BASE}/events/{match['match_id']}.json")
    by_id = {e["id"]: e for e in events}
    out = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        kp_id = e.get("shot", {}).get("key_pass_id")
        out.append({
            "match_id": match["match_id"],
            "shot": e,
            "key_pass": by_id.get(kp_id) if kp_id else None,
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cid, sid, label in COMPETITIONS:
        path = os.path.join(OUT_DIR, f"{cid}_{sid}.jsonl")
        if os.path.exists(path):
            print(f"skip {label} (exists)")
            continue
        matches = get_json(f"{BASE}/matches/{cid}/{sid}.json")
        rows = []
        with ThreadPoolExecutor(max_workers=16) as pool:
            futs = [pool.submit(shots_for_match, m) for m in matches]
            for f in as_completed(futs):
                rows.extend(f.result())
        meta = {m["match_id"]: m for m in matches}
        with open(path + ".tmp", "w") as fh:
            for r in rows:
                m = meta[r["match_id"]]
                r.update({
                    "competition_id": cid, "season_id": sid, "competition": label,
                    "match_date": m["match_date"],
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                })
                fh.write(json.dumps(r) + "\n")
        os.replace(path + ".tmp", path)
        print(f"{label}: {len(matches)} matches, {len(rows)} shots", flush=True)


if __name__ == "__main__":
    sys.exit(main())
