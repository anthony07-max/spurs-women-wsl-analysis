"""
Re-train the xG model from github.com/anthony07-max/football-xg-model and apply it
to new shots.

The features, filters, LightGBM settings and early-stopping split are copied from
that repo's notebook (xg_model.ipynb), so the model here is the same model: it is
trained on ~57,700 men's non-penalty shots from StatsBomb Open Data. No women's
shots are used in training.

    from xg_model import train_xg_model, predict_xg
    model = train_xg_model("data/train_shots")
    wsl["xg"] = predict_xg(model, wsl)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

from features import load_shots

SEED = 42
NUM = ["distance", "log_distance", "angle", "x", "y", "n_def_in_cone", "n_opp_within_3",
       "gk_dist_to_goal", "gk_in_cone"]
BIN = ["under_pressure", "first_time", "one_on_one", "open_goal", "follows_dribble",
       "aerial_won", "has_freeze_frame"]
CAT = ["body_group", "shot_type", "technique", "play_pattern", "assist_type"]
FEATURES = NUM + BIN + CAT
LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=80,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=3000, random_state=SEED, verbose=-1)


def prepare(shots, categories=None):
    """Non-penalty, non-shoot-out shots with model features. `categories` fixes the
    category levels to those seen in training so new data is encoded identically."""
    df = shots[(shots.shot_type != "Penalty") & (shots.period != 5)].reset_index(drop=True).copy()
    df["log_distance"] = np.log1p(df["distance"])
    for c in CAT:
        df[c] = df[c].astype("category" if categories is None else pd.CategoricalDtype(categories[c]))
    return df


def train_xg_model(train_dir):
    """Reproduce the football-xg-model LightGBM and return a dict with the fitted
    model, its held-out test metrics (to check the reproduction) and the in-game
    penalty value."""
    # The download writes matches in completion order, so sort first: LightGBM's row
    # subsampling depends on row order, and this makes every run give the same model.
    shots = (load_shots(train_dir)
             .sort_values(["competition_id", "season_id", "match_id", "period", "minute", "shot_id"])
             .reset_index(drop=True))
    pens = shots[(shots.shot_type == "Penalty") & (shots.period != 5)]
    df = prepare(shots)
    y = df.is_goal.values

    # Same splits as the original notebook: 20% of matches held out for testing,
    # then 15% of the remaining matches for early stopping.
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(df, y, df.match_id))
    train, test = df.iloc[tr_i], df.iloc[te_i]
    f_i, v_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED)
                    .split(train, train.is_goal, train.match_id))
    fit, val = train.iloc[f_i], train.iloc[v_i]
    gbm = lgb.LGBMClassifier(**LGB_PARAMS)
    gbm.fit(fit[FEATURES], fit.is_goal, eval_set=[(val[FEATURES], val.is_goal)],
            eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
    best = gbm.best_iteration_
    p = gbm.predict_proba(test[FEATURES], num_iteration=best)[:, 1]
    test_metrics = dict(shots=len(test), matches=test.match_id.nunique(),
                        log_loss=log_loss(test.is_goal, p), brier=brier_score_loss(test.is_goal, p),
                        roc_auc=roc_auc_score(test.is_goal, p),
                        sb_log_loss=log_loss(test.is_goal, test.statsbomb_xg),
                        sb_roc_auc=roc_auc_score(test.is_goal, test.statsbomb_xg))

    # Final model for scoring new shots: all men's shots, same number of trees.
    final = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": best}).fit(df[FEATURES], y)
    return dict(model=final, best_iteration=best, test_metrics=test_metrics,
                n_train_shots=len(df), n_train_matches=df.match_id.nunique(),
                categories={c: list(df[c].cat.categories) for c in CAT},
                pen_xg=float(pens.is_goal.mean()))


def predict_xg(bundle, shots):
    """xG for every shot in `shots` (penalties get the in-game penalty conversion rate
    from the training data). Returns a Series aligned to `shots`."""
    out = pd.Series(np.nan, index=shots.index, dtype=float)
    pen = (shots.shot_type == "Penalty")
    out[pen] = bundle["pen_xg"]
    np_mask = ~pen & (shots.period != 5)
    X = shots.loc[np_mask].copy()
    X["log_distance"] = np.log1p(X["distance"])
    for c in CAT:
        X[c] = X[c].astype(pd.CategoricalDtype(bundle["categories"][c]))
    out[np_mask] = bundle["model"].predict_proba(X[FEATURES])[:, 1]
    return out
