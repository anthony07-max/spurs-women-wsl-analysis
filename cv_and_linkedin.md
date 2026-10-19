# CV bullets and LinkedIn post: Spurs Women WSL analysis

All figures below come from the executed notebook (`spurs_wsl.ipynb`) and `results/`.

## CV bullet points (choose 2-3)

**Spurs Women FA WSL Analysis | Python, LightGBM, pandas, mplsoccer** · github.com/anthony07-max/spurs-women-wsl-analysis

- Built team-level chance creation and prevention metrics (non-penalty xG, xG per shot, key passes, shot-creating actions, progressive passes) with league rankings for every FA WSL team across 457 matches of StatsBomb event data, showing Tottenham Hotspur Women's xG difference rising from −0.89 a match (11th of 12) in 2019/20 to +0.10 (5th) in 2023/24.
- Applied my men's-trained xG model to 11,711 WSL shots. It predicted 1,279.4 xG for 1,300 goals (StatsBomb's own xG: 1,230.0) with log loss 0.2915 against 0.2951, and I identified where it fails: it over-rates big chances in the women's game.
- Produced position-aware per-90 percentile radars for six 2023/24 Spurs players against 487 WSL player-seasons, with possession-adjusted defending and small-sample shrinkage.

*Shorter one-line version:* Applied my xG model and player-profile method to Spurs Women's WSL seasons (StatsBomb Open Data, 2019/20 to 2023/24), benchmarking against StatsBomb xG and ranking Spurs' chance creation and prevention against the league.

## LinkedIn post (draft)

I took my xG model, trained only on men's football, and pointed it at the Women's Super League.

Using StatsBomb's free WSL data, I looked at Tottenham Hotspur Women in 2023/24 and compared it with their first two top-flight seasons:

- 2023/24 was their best season in the data: +0.10 xG difference a match, 5th of 12 and behind only the big four. In 2019/20 it was −0.89 (11th).
- The jump came from creation: 8.7 key passes a match (4th), up from 4.4 in 2020/21.
- Shot quality still trails the league: 0.105 xG per shot (7th), with 63% of shots from inside the box against 72% for everyone else.
- Grace Clinton drove a lot of it: 3.48 shot-creating actions per 90 and a 71% dribble success rate, 97th percentile among WSL attacking midfielders and wingers.
- The men's model held up: 1,279.4 xG for 1,300 goals on 11,711 WSL shots. It does over-rate the biggest chances, which is the women's game telling me something.

2023/24 is the newest season in the open data, so this is a method applied to Spurs Women rather than a view of the current team. Code, charts and limitations are on GitHub (link in comments).

#WSL #SpursWomen #FootballAnalytics #DataScience #Python
