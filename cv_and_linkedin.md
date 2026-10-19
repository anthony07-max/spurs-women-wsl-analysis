# CV bullets and LinkedIn post: Spurs Women WSL analysis

All figures below come from the executed notebook (`spurs_wsl.ipynb`) and `results/`.

## CV bullet points (choose 2-3)

**Spurs Women FA WSL Analysis | Python, LightGBM, pandas, mplsoccer** · github.com/anthony07-max/spurs-women-wsl-analysis

- Applied my men's-trained xG model to 8,205 FA WSL shots from StatsBomb Open Data (325 matches, 2018/19 to 2020/21). It predicted 895.4 xG for 898 goals and matched StatsBomb's own xG (log loss 0.2881 vs 0.2931, ROC AUC 0.784 vs 0.774), and I identified where it fails (it over-rates big chances in the women's game).
- Built team-level chance creation and prevention metrics (non-penalty xG, xG per shot, key passes, shot-creating actions, progressive passes) and league rankings for every WSL team, showing that Tottenham Hotspur Women cut the xG they conceded from 1.78 to 1.43 a match (10th to 6th of 12) between 2019/20 and 2020/21.
- Produced position-aware per-90 percentile radars for six Spurs players against 347 WSL player-seasons, with possession-adjusted defending and small-sample shrinkage.

*Shorter one-line version:* Applied my xG model and player-profile method to Spurs Women's WSL seasons (StatsBomb Open Data), benchmarking against StatsBomb xG and ranking Spurs' chance creation and prevention against the league.

## LinkedIn post (draft)

I took my xG model, trained only on men's football, and pointed it at the Women's Super League.

Using StatsBomb's free WSL data (2018/19 to 2020/21), I looked at Tottenham Hotspur Women's first two top-flight seasons:

- The men's model held up: 895.4 xG for 898 goals on 8,205 WSL shots, slightly ahead of StatsBomb's own xG on log loss. It does over-rate the biggest chances, which is the women's game telling me something.
- Spurs got much harder to create against: 1.78 xG conceded a match in 2019/20 (10th of 12), 1.43 in 2020/21 (6th).
- The attacking issue was shot quality, not volume: 0.074 xG per shot in 2019/20, 11th in the league, with half their shots from outside the box.
- Kit Graham was the creative hub: 1.77 key passes per 90, 89th percentile among WSL attacking midfielders and wingers.

It's older open data, so it's a method applied to Spurs Women rather than a view of the current team. Code, charts and limitations are on GitHub (link in comments).

#WSL #SpursWomen #FootballAnalytics #DataScience #Python
