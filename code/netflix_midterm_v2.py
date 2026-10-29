"""Kalshi Netflix US movie weekly-winner prediction, midterm v2 (exploratory re-analysis).

Run:  python netflix_midterm_v2.py --data-dir ../input_data --output-dir ../results --figure-dir ../figures
Local files only; no credentials, no API calls.
Inputs:   markets_and_results.csv, hourly_candles.parquet
          (if pyarrow is not installed, hourly_candles_csv_mirror.csv is used and this is recorded in run_record.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261022
EPS = 1e-6                       # probabilities are clipped to [EPS, 1 - EPS] before log loss
TIE_TOL = 1e-6                   # validation log-loss ties (see PROTOCOL_v2.md)
CUTOFFS = [24, 72]               # 24 h = primary, 72 h = exploratory
MARKET = "Market leader"
LR_METHODS = ["A: Price", "B: Price + change", "C: Price + change + volume"]
RF_METHOD = "RF: full features"
METHODS = [MARKET] + LR_METHODS + [RF_METHOD]
FEATURES = {
    LR_METHODS[0]: ["price_now"],
    LR_METHODS[1]: ["price_now", "price_change_24h"],
    LR_METHODS[2]: ["price_now", "price_change_24h", "log_volume_24h"],
    RF_METHOD: ["price_now", "price_change_24h", "log_volume_24h"],
}
LR_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RF_GRID = [(d, leaf) for d in (2, 3, 5) for leaf in (5, 10, 20)]
RF_TREES = 300


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- data
def load_inputs(data_dir):
    data_dir = Path(data_dir)
    markets = pd.read_csv(data_dir / "markets_and_results.csv", dtype=str, keep_default_na=False)
    source = "parquet"
    try:
        candles = pd.read_parquet(data_dir / "hourly_candles.parquet")
    except ImportError:
        candles = pd.read_csv(data_dir / "hourly_candles_csv_mirror.csv")
        source = "csv_mirror_of_parquet"
    return markets, candles, source


def build_features(markets, candles, cutoff_hours, split_map=None):
    """Features use only candles that end at or before the cutoff; labels are never features."""
    m = markets.copy()
    for column in ["open_time", "close_time", "occurrence_datetime", "settlement_ts"]:
        # format="ISO8601": the timestamps mix fractional-second lengths (one open_time has milliseconds);
        # inferring the format from the first row would silently turn such rows into NaT.
        m[column + "_dt"] = pd.to_datetime(m[column].replace("", None), utc=True, format="ISO8601")
    assert m.open_time_dt.notna().all() and m.close_time_dt.notna().all(), "Unparsed open/close time"
    c = candles.sort_values(["market_ticker", "end_period_ts"]).copy()
    for col in ["price_close", "volume", "end_period_ts"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    assert not m.market_ticker.duplicated().any(), "Duplicate contract metadata"
    assert not c.duplicated(["market_ticker", "end_period_ts"]).any(), "Duplicate candle keys"
    assert c.price_close.dropna().between(0, 1).all(), "Invalid normalized price"
    assert (c.volume.dropna() >= 0).all(), "Negative volume"
    candle_groups = {k: g for k, g in c.groupby("market_ticker", sort=False)}
    audit, rows = [], []
    for event, group in m.groupby("event_ticker", sort=False):
        yes = group[group.result == "yes"]
        reason = "included"
        if not group.result.isin(["yes", "no"]).all():
            reason = "not_all_settled"
        elif len(yes) != 1:
            reason = "not_exactly_one_yes"
        elif group.close_time_dt.nunique() != 1:
            reason = "no_common_close_time"
        if reason != "included":
            audit.append(dict(event_ticker=event, status=reason, all_candidates=len(group),
                              yes_count=len(yes), eligible_candidates=0))
            continue
        close = group.close_time_dt.iloc[0]
        t = (close - pd.Timedelta(hours=cutoff_hours)).floor("h")
        t_int = int(t.timestamp())
        past_int = t_int - 86400
        eligible = group[group.open_time_dt <= t].sort_values("market_ticker")
        winner = yes.iloc[0]
        winner_listed = bool((eligible.market_ticker == winner.market_ticker).any())
        rule_dates = group.rules_primary.str.extract(r"\bon ([A-Za-z]+ \d{1,2}, \d{4})", expand=False)
        assert rule_dates.notna().all() and rule_dates.nunique() == 1, f"Publication rule differs: {event}"
        rule_date = datetime.strptime(rule_dates.iloc[0], "%b %d, %Y").date()
        assert t.date() < rule_date, f"Prediction not before publication date in rules: {event}"
        audit.append(dict(
            event_ticker=event, status=reason, all_candidates=len(group), yes_count=1,
            eligible_candidates=len(eligible), prediction_time_utc=t.isoformat(),
            close_time_utc=close.isoformat(), winner_ticker=winner.market_ticker,
            winner_movie=winner.movie, winner_listed_at_prediction=winner_listed,
            occurrence_datetime=winner.occurrence_datetime, rule_publication_date=rule_date.isoformat(),
            winner_settlement_time=winner.settlement_ts))
        for _, movie in eligible.iterrows():
            h = candle_groups.get(movie.market_ticker, c.iloc[0:0])
            prior = h[(h.end_period_ts <= t_int) & (h.end_period_ts >= movie.open_time_dt.timestamp())]
            priced = prior[prior.price_close.notna()]
            old = priced[priced.end_period_ts <= past_int]
            window = prior[(prior.end_period_ts > past_int) & (prior.end_period_ts <= t_int)]
            latest = priced.iloc[-1] if len(priced) else None
            previous = old.iloc[-1] if len(old) else None
            p_now = float(latest.price_close) if latest is not None else np.nan
            p_old = float(previous.price_close) if previous is not None else np.nan
            volume = float(window.volume.sum(min_count=1)) if len(window) else np.nan
            last_ts = float(latest.end_period_ts) if latest is not None else np.nan
            old_ts = float(previous.end_period_ts) if previous is not None else np.nan
            rows.append(dict(
                cutoff_hours=cutoff_hours, event_ticker=event, market_ticker=movie.market_ticker,
                movie=movie.movie, prediction_time_utc=t.isoformat(), prediction_ts=t_int,
                price_now=p_now, price_24h_ago=p_old,
                price_change_24h=p_now - p_old if pd.notna(p_now) and pd.notna(p_old) else np.nan,
                volume_24h=volume, log_volume_24h=np.log1p(volume) if pd.notna(volume) else np.nan,
                price_age_hours=(t_int - last_ts) / 3600 if pd.notna(last_ts) else np.nan,
                returned_candle_hours_24h=len(window),
                traded_hours_24h=int((window.volume > 0).sum()),
                price_source_ts=last_ts, old_price_source_ts=old_ts,
                label_yes=int(movie.result == "yes")))
    features, audit = pd.DataFrame(rows), pd.DataFrame(audit)
    included = audit[audit.status == "included"].sort_values(["prediction_time_utc", "event_ticker"])
    if split_map is None:
        n = len(included)
        n_train, n_valid = int(n * .6), int(n * .2)
        split_map = {e: ("train" if i < n_train else "validation" if i < n_train + n_valid else "test")
                     for i, e in enumerate(included.event_ticker)}
    features["split"] = features.event_ticker.map(split_map)
    audit["split"] = audit.event_ticker.map(split_map).fillna("excluded")
    features = features.sort_values(["prediction_ts", "event_ticker", "market_ticker"]).reset_index(drop=True)
    audit = audit.sort_values(["prediction_time_utc", "event_ticker"], na_position="last").reset_index(drop=True)
    ok = features.price_source_ts.notna()
    assert (features.loc[ok, "price_source_ts"] <= features.loc[ok, "prediction_ts"]).all()
    ok = features.old_price_source_ts.notna()
    assert (features.loc[ok, "old_price_source_ts"] <= features.loc[ok, "prediction_ts"] - 86400).all()
    assert not features.duplicated(["event_ticker", "market_ticker"]).any()
    assert features.groupby("event_ticker").split.nunique().max() == 1
    occurrence = pd.to_datetime(audit.loc[audit.status == "included", "occurrence_datetime"], utc=True, format="ISO8601")
    known = occurrence.notna()
    pred_t = pd.to_datetime(audit.loc[audit.status == "included", "prediction_time_utc"], utc=True, format="ISO8601")
    assert (occurrence[known] > pred_t[known]).all(), "Prediction after scheduled result"
    for earlier, later in [("train", "validation"), ("validation", "test")]:
        assert (pd.to_datetime(audit.loc[audit.split == earlier, "winner_settlement_time"], utc=True, format="ISO8601").max() <
                pd.to_datetime(audit.loc[audit.split == later, "prediction_time_utc"], utc=True, format="ISO8601").min())
    return features, audit, split_map


# ----------------------------------------------------------------------------- metrics
def clipped(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def per_week_metric(df, prob_col, metric):
    """Mean loss within each week over scorable candidates (observed market price). Returns a Series by week."""
    p, y = clipped(df[prob_col]), df.label_yes.to_numpy()
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)) if metric == "log_loss" else (p - y) ** 2
    return pd.Series(loss, index=df.index).groupby(df.event_ticker).mean()


def week_equal(df, prob_col, metric):
    s = per_week_metric(df, prob_col, metric)
    return float(s.mean()) if len(s) else np.nan


def make_pipeline(kind, **params):
    if kind == "lr":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=params["C"], l1_ratio=0.0, solver="lbfgs",
                                         max_iter=5000, tol=1e-8, random_state=SEED))])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", RandomForestClassifier(n_estimators=RF_TREES, max_depth=params["max_depth"],
                                         min_samples_leaf=params["min_samples_leaf"], max_features=None,
                                         random_state=SEED, n_jobs=1))])


def week_weights(train):
    return len(train) / (train.event_ticker.nunique() * train.groupby("event_ticker").market_ticker.transform("size"))


def fit_predict(train, rows, columns, kind, **params):
    model = make_pipeline(kind, **params)
    model.fit(train[columns], train.label_yes, model__sample_weight=week_weights(train))
    return model, model.predict_proba(rows[columns])[:, 1]


def select_by_validation(rows):
    best_loss = min(r["val_log_loss"] for r in rows)
    tied = [r for r in rows if r["val_log_loss"] - best_loss < TIE_TOL]
    return min(tied, key=lambda r: r["tie_key"])


def weekly_top1(data, audit, methods):
    truth = audit[audit.status == "included"].set_index("event_ticker")
    out = []
    for event, tr in truth.iterrows():
        group = data[data.event_ticker == event]
        for method in methods:
            predicted, tied = None, 0
            if len(group) and not (method == MARKET and group.price_now.notna().sum() == 0):
                score = group.price_now.where(group.price_now.notna(), -np.inf) if method == MARKET else group[method]
                top = score.max()
                best = group.assign(_s=score).sort_values(["_s", "market_ticker"], ascending=[False, True]).iloc[0]
                predicted, tied = best.market_ticker, int((score == top).sum())
            out.append(dict(cutoff_hours=int(group.cutoff_hours.iloc[0]) if len(group) else np.nan,
                            event_ticker=event, split=tr.split, method=method,
                            prediction_time_utc=tr.prediction_time_utc,
                            predicted_ticker=predicted, winner_ticker=tr.winner_ticker,
                            winner_movie=tr.winner_movie, correct=int(predicted == tr.winner_ticker),
                            tied_best_candidates=tied, candidates_listed=len(group),
                            winner_listed_at_prediction=bool(tr.winner_listed_at_prediction)))
    return pd.DataFrame(out)


def bootstrap_diff(values, rng, n_boot=10000):
    values = np.asarray(values, dtype=float)
    boot = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return np.quantile(boot, [.025, .975])


def evaluate(data, predictions, cutoff):
    """Weekly top-1 plus candidate-level probability metrics on the same scorable rows for every method."""
    rows, paired = [], []
    prob_methods = {"Market price (contract probability)": "price_now"}
    prob_methods.update({m: m for m in LR_METHODS + [RF_METHOD]})
    for split in ["train", "validation", "test"]:
        part = data[data.split == split]
        scorable = part[part.price_now.notna()]
        n_weeks = part.event_ticker.nunique()
        top = predictions[predictions.split == split]
        for method in METHODS:
            g = top[top.method == method]
            k, n = int(g.correct.sum()), len(g)
            lo, hi = wilson(k, n)
            row = dict(cutoff_hours=cutoff, split=split, method=method, top1_correct=k, weeks=n,
                       top1_accuracy=k / n, wilson_low=lo, wilson_high=hi)
            label = {MARKET: "Market price (contract probability)"}.get(method, method)
            if label in prob_methods:
                col = prob_methods[label]
                row.update(log_loss_week_equal=week_equal(scorable, col, "log_loss"),
                           brier_week_equal=week_equal(scorable, col, "brier"),
                           log_loss_pooled=float(np.mean(-(scorable.label_yes * np.log(clipped(scorable[col])) +
                                                           (1 - scorable.label_yes) * np.log(1 - clipped(scorable[col]))))),
                           brier_pooled=float(np.mean((clipped(scorable[col]) - scorable.label_yes) ** 2)),
                           scored_rows=len(scorable), total_rows=len(part),
                           scored_weeks=scorable.event_ticker.nunique(), total_weeks=n_weeks,
                           clipped_rows=int(((scorable[col] < EPS) | (scorable[col] > 1 - EPS)).sum()))
            rows.append(row)
    rng = np.random.default_rng(SEED)
    comparisons = [(LR_METHODS[1], LR_METHODS[0]), (LR_METHODS[2], LR_METHODS[1]),
                   (LR_METHODS[2], MARKET), (RF_METHOD, MARKET), (RF_METHOD, LR_METHODS[2])]
    for split in ["validation", "test"]:
        wide = predictions[predictions.split == split].pivot(index="event_ticker", columns="method", values="correct")
        part = data[(data.split == split) & data.price_now.notna()]
        for newer, older in comparisons:
            diff = (wide[newer] - wide[older]).to_numpy()
            lo, hi = bootstrap_diff(diff, rng)
            paired.append(dict(cutoff_hours=cutoff, split=split, metric="top1_accuracy", newer=newer, older=older,
                               weeks=len(diff), difference=float(diff.mean()), boot_low=float(lo), boot_high=float(hi),
                               newer_only_correct=int((diff == 1).sum()), older_only_correct=int((diff == -1).sum())))
            for metric in ["log_loss", "brier"]:
                a = per_week_metric(part, newer if newer != MARKET else "price_now", metric)
                b = per_week_metric(part, older if older != MARKET else "price_now", metric)
                d = (a - b).dropna().to_numpy()
                lo, hi = bootstrap_diff(d, rng)
                paired.append(dict(cutoff_hours=cutoff, split=split, metric=metric + "_week_equal", newer=newer, older=older,
                                   weeks=len(d), difference=float(d.mean()), boot_low=float(lo), boot_high=float(hi),
                                   newer_only_correct=np.nan, older_only_correct=np.nan))
    return pd.DataFrame(rows), pd.DataFrame(paired)


def calibration_table(df, col, label, cutoff, sample):
    bins = np.linspace(0, 1, 6)
    d = df[df[col].notna()].copy()
    d["bin"] = pd.cut(d[col], bins=bins, include_lowest=True, right=True)
    g = d.groupby("bin", observed=False)
    out = pd.DataFrame({"candidate_weeks": g.size(), "weeks_in_bin": g.event_ticker.nunique(),
                        "mean_predicted": g[col].mean(), "observed_win_rate": g.label_yes.mean()}).reset_index()
    out["bin"] = out["bin"].astype(str)
    out.insert(0, "sample", sample)
    out.insert(0, "source", label)
    out.insert(0, "cutoff_hours", cutoff)
    return out


# ----------------------------------------------------------------------------- one cutoff
def run_cutoff(markets, candles, cutoff, split_map):
    features, audit, split_map = build_features(markets, candles, cutoff, split_map)
    data = features.copy()
    train, valid = data[data.split == "train"], data[data.split == "validation"]
    grid, chosen, models = [], {}, {}
    valid_scorable = valid[valid.price_now.notna()]
    for method in LR_METHODS:
        rows = []
        for c_value in LR_GRID:
            _, p = fit_predict(train, valid, FEATURES[method], "lr", C=c_value)
            v = valid.assign(_p=p)
            vs = v[v.price_now.notna()]
            picked = v.sort_values(["_p", "market_ticker"], ascending=[False, True]).groupby("event_ticker").head(1)
            rows.append(dict(cutoff_hours=cutoff, model=method, param_C=c_value, param_max_depth=np.nan,
                             param_min_samples_leaf=np.nan, val_log_loss=week_equal(vs, "_p", "log_loss"),
                             val_brier=week_equal(vs, "_p", "brier"), val_top1_correct=int(picked.label_yes.sum()),
                             val_weeks=v.event_ticker.nunique(), tie_key=(c_value,)))
        grid += rows
        chosen[method] = select_by_validation(rows)
    rows = []
    for depth, leaf in RF_GRID:
        _, p = fit_predict(train, valid, FEATURES[RF_METHOD], "rf", max_depth=depth, min_samples_leaf=leaf)
        v = valid.assign(_p=p)
        vs = v[v.price_now.notna()]
        picked = v.sort_values(["_p", "market_ticker"], ascending=[False, True]).groupby("event_ticker").head(1)
        rows.append(dict(cutoff_hours=cutoff, model=RF_METHOD, param_C=np.nan, param_max_depth=depth,
                         param_min_samples_leaf=leaf, val_log_loss=week_equal(vs, "_p", "log_loss"),
                         val_brier=week_equal(vs, "_p", "brier"), val_top1_correct=int(picked.label_yes.sum()),
                         val_weeks=v.event_ticker.nunique(), tie_key=(depth, -leaf)))
    grid += rows
    chosen[RF_METHOD] = select_by_validation(rows)
    grid = pd.DataFrame(grid)
    grid["selected"] = False
    for method, r in chosen.items():
        mask = (grid.model == method)
        if method == RF_METHOD:
            mask &= (grid.param_max_depth == r["param_max_depth"]) & (grid.param_min_samples_leaf == r["param_min_samples_leaf"])
        else:
            mask &= (grid.param_C == r["param_C"])
        grid.loc[mask, "selected"] = True
    grid["tie_key"] = grid.tie_key.astype(str)
    importances, coefficients = [], []
    for method in LR_METHODS + [RF_METHOD]:
        r = chosen[method]
        if method == RF_METHOD:
            model, p = fit_predict(train, data, FEATURES[method], "rf", max_depth=int(r["param_max_depth"]),
                                   min_samples_leaf=int(r["param_min_samples_leaf"]))
            names = model.named_steps["imputer"].get_feature_names_out(FEATURES[method])
            for name, imp in zip(names, model.named_steps["model"].feature_importances_):
                importances.append(dict(cutoff_hours=cutoff, feature=name, impurity_importance=float(imp)))
        else:
            model, p = fit_predict(train, data, FEATURES[method], "lr", C=float(r["param_C"]))
            names = model.named_steps["imputer"].get_feature_names_out(FEATURES[method])
            for name, coef in zip(names, model.named_steps["model"].coef_[0]):
                coefficients.append(dict(cutoff_hours=cutoff, method=method, feature=name, standardized_coefficient=float(coef)))
        data[method] = p
        models[method] = model
    predictions = weekly_top1(data, audit, METHODS)
    metrics, paired = evaluate(data, predictions, cutoff)
    return dict(features=features, audit=audit, data=data, grid=grid, chosen=chosen, predictions=predictions,
                metrics=metrics, paired=paired, importances=pd.DataFrame(importances),
                coefficients=pd.DataFrame(coefficients), split_map=split_map)


def common_week_table(res):
    """Compare cutoffs on weeks whose winner was listed at both cutoffs (test weeks)."""
    listed = {c: res[c]["audit"].set_index("event_ticker").winner_listed_at_prediction.dropna().astype(bool) for c in CUTOFFS}
    common = [e for e in listed[24].index if bool(listed[24][e]) and bool(listed[72].get(e, False))]
    rows = []
    for c in CUTOFFS:
        pred = res[c]["predictions"]
        data = res[c]["data"]
        for split in ["validation", "test"]:
            for sample, weeks in [("full sample", None), ("common weeks", common)]:
                p = pred[pred.split == split]
                d = data[(data.split == split) & data.price_now.notna()]
                if weeks is not None:
                    p, d = p[p.event_ticker.isin(weeks)], d[d.event_ticker.isin(weeks)]
                for method in METHODS:
                    g = p[p.method == method]
                    row = dict(cutoff_hours=c, split=split, sample=sample, method=method, top1_correct=int(g.correct.sum()),
                               weeks=len(g), top1_accuracy=float(g.correct.mean()) if len(g) else np.nan)
                    col = "price_now" if method == MARKET else method
                    if col in d.columns and len(d):
                        row.update(log_loss_week_equal=week_equal(d, col, "log_loss"),
                                   brier_week_equal=week_equal(d, col, "brier"), scored_rows=len(d))
                    rows.append(row)
    return pd.DataFrame(rows), common


def describe(res):
    rows = []
    for c in CUTOFFS:
        f, a = res[c]["features"], res[c]["audit"]
        inc = a[a.status == "included"]
        rows += [dict(cutoff_hours=c, item="included weeks", value=len(inc)),
                 dict(cutoff_hours=c, item="eligible candidate-week rows", value=len(f)),
                 dict(cutoff_hours=c, item="median candidates per week", value=float(inc.eligible_candidates.median())),
                 dict(cutoff_hours=c, item="min candidates per week", value=int(inc.eligible_candidates.min())),
                 dict(cutoff_hours=c, item="max candidates per week", value=int(inc.eligible_candidates.max())),
                 dict(cutoff_hours=c, item="weeks with no eligible candidate", value=int((inc.eligible_candidates == 0).sum())),
                 dict(cutoff_hours=c, item="weeks where winner not listed", value=int((~inc.winner_listed_at_prediction.astype(bool)).sum())),
                 dict(cutoff_hours=c, item="rows without observed price", value=int(f.price_now.isna().sum())),
                 dict(cutoff_hours=c, item="rows with price older than 24h", value=int((f.price_age_hours > 24).sum())),
                 dict(cutoff_hours=c, item="rows without 24h price change", value=int(f.price_change_24h.isna().sum())),
                 dict(cutoff_hours=c, item="rows with no returned hourly row in 24h window", value=int(f.volume_24h.isna().sum())),
                 dict(cutoff_hours=c, item="rows with returned window but zero volume", value=int((f.volume_24h == 0).sum())),
                 dict(cutoff_hours=c, item="rows with positive 24h volume", value=int((f.volume_24h > 0).sum())),
                 dict(cutoff_hours=c, item="weeks with no observed price at all", value=int(f.groupby("event_ticker").price_now.count().eq(0).sum())),
                 dict(cutoff_hours=c, item="median observed price_now", value=float(f.price_now.median())),
                 dict(cutoff_hours=c, item="share of rows with price_now < 0.10", value=float((f.price_now.dropna() < .10).mean())),
                 dict(cutoff_hours=c, item="winner rows with observed price", value=int(f[(f.label_yes == 1)].price_now.notna().sum()))]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def market_all_weeks_table(res, common_weeks):
    """Descriptive only: the market-leader rule and market price over all included weeks (train + validation + test)."""
    rows = []
    for c in CUTOFFS:
        pred, data = res[c]["predictions"], res[c]["data"]
        for sample, weeks in [("all included weeks", None), ("weeks listed at both cutoffs", common_weeks)]:
            p = pred[pred.method == MARKET]
            d = data[data.price_now.notna()]
            if weeks is not None:
                p, d = p[p.event_ticker.isin(weeks)], d[d.event_ticker.isin(weeks)]
            k, n = int(p.correct.sum()), len(p)
            lo, hi = wilson(k, n)
            rows.append(dict(cutoff_hours=c, sample=sample, weeks=n, market_top1_correct=k, market_top1_accuracy=k / n,
                             wilson_low=lo, wilson_high=hi, scored_rows=len(d),
                             market_log_loss_week_equal=week_equal(d, "price_now", "log_loss"),
                             market_brier_week_equal=week_equal(d, "price_now", "brier")))
    return pd.DataFrame(rows)


def figures(res, common_table, common_weeks, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    colors = {MARKET: "#6b7785", LR_METHODS[0]: "#486581", LR_METHODS[1]: "#0b7285", LR_METHODS[2]: "#d9480f", RF_METHOD: "#5f3dc4"}
    short = {MARKET: "Highest\nprice", LR_METHODS[0]: "LR-A\nprice", LR_METHODS[1]: "LR-B\n+ change",
             LR_METHODS[2]: "LR-C\n+ change\n+ volume", RF_METHOD: "RF\nfull\nfeatures"}
    r24 = res[24]
    t = r24["metrics"].query("split == 'test'").set_index("method").loc[METHODS]
    short3 = {MARKET: "Market\nleader", LR_METHODS[0]: "LR-A", LR_METHODS[1]: "LR-B", LR_METHODS[2]: "LR-C", RF_METHOD: "RF"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    ax = axes[0]
    y = t.top1_accuracy.to_numpy() * 100
    err = np.vstack([y - t.wilson_low.to_numpy() * 100, t.wilson_high.to_numpy() * 100 - y])
    ax.bar(range(5), y, color=[colors[x] for x in METHODS], width=.8)
    ax.errorbar(range(5), y, yerr=err, fmt="none", ecolor="#333", capsize=4)
    for i, row in enumerate(t.itertuples()):
        ax.text(i, 6, f"{row.top1_correct}/{row.weeks}", ha="center", color="white", fontweight="bold", fontsize=11.5)
    ax.set(xticks=range(5), xticklabels=[short3[x] for x in METHODS], ylim=(0, 104), ylabel="Weekly top-1 accuracy (%)",
           title=f"Top-1: weeks correct / {int(t.weeks.iloc[0])} test weeks\n(whiskers: Wilson 95%)")
    for ax, col, title in [(axes[1], "log_loss_week_equal", "Binary log loss (lower is better)"),
                           (axes[2], "brier_week_equal", "Brier score (lower is better)")]:
        vals = [t.loc[x, col] for x in METHODS]
        ax.bar(range(5), vals, color=[colors[x] for x in METHODS], width=.8)
        ax.axhline(vals[0], color="#333", linestyle="--", linewidth=1)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=12)
        ax.set(xticks=range(5), xticklabels=[short3[x] for x in METHODS], title=title + "\ndashed line = market level",
               ylim=(0, max(vals) * 1.15))
    for ax in axes:
        ax.tick_params(axis="x", labelsize=12.5)
        ax.tick_params(axis="y", labelsize=12)
        ax.title.set_fontsize(13)
        ax.yaxis.label.set_fontsize(12.5)
    fig.text(.5, .012, "LR-A: price   |   LR-B: + 24h change   |   LR-C: + log volume   |   RF: all three features   |   "
             "Market: highest price (probability = contract's own price)   |   24h cutoff",
             ha="center", fontsize=11, color="#444")
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(out / "01_model_comparison_24h.png", bbox_inches="tight")
    plt.close(fig)

    f = r24["features"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(f.price_now.dropna(), bins=20, range=(0, 1), color="#486581")
    ax.set(title="Last observed price, all candidate-week rows (24h cutoff)", xlabel="Contract price ($)", ylabel="Candidate-week rows")
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "02_market_description_24h.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ax.hist(f.traded_hours_24h, bins=np.arange(-.5, 25.5), color="#d9480f")
    ax.set(title="Hours with trades in the last 24h (24h cutoff)", xlabel="Traded hours (of 24)", ylabel="Candidate-week rows")
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(labelsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "02b_trading_hours_24h.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    labels = ["<= 6h old", "6-24h old", "> 24h old", "No price"]
    for ax, c in zip(axes, CUTOFFS):
        age = res[c]["features"].price_age_hours
        counts = [int((age <= 6).sum()), int(((age > 6) & (age <= 24)).sum()), int((age > 24).sum()), int(age.isna().sum())]
        ax.barh(labels, counts, color=["#0b7285", "#94a3b8", "#e9a23b", "#b91c1c"])
        for i, v in enumerate(counts):
            ax.text(v + 5, i, str(v), va="center", fontsize=11)
        ax.set_title(f"Age of last price, {c}h cutoff", fontsize=11.5)
        ax.set_xlabel("Candidate-week rows", fontsize=11)
        ax.set_xlim(0, max(counts) * 1.25)
        ax.tick_params(labelsize=11)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "03_price_freshness_by_cutoff.png", bbox_inches="tight")
    plt.close(fig)

    cal = res["calibration"]
    edges = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
    markers = {"Market price": "o", LR_METHODS[2]: "s", RF_METHOD: "^"}
    mcolors = {"Market price": "#6b7785", LR_METHODS[2]: "#d9480f", RF_METHOD: "#5f3dc4"}
    for c in CUTOFFS:
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.7))
        fig.subplots_adjust(wspace=.26, bottom=.38, top=.85, left=.08, right=.99)
        for j, sample in enumerate(["all included weeks (market only)", "test weeks"]):
            ax = axes[j]
            ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
            sources = ["Market price"] if j == 0 else ["Market price", LR_METHODS[2], RF_METHOD]
            cells, source_handles = [], []
            for source in sources:
                d = cal[(cal.cutoff_hours == c) & (cal["sample"] == sample) & (cal.source == source)].reset_index(drop=True)
                cells.append([str(int(n)) for n in d.candidate_weeks])
                d = d[d.candidate_weeks > 0]
                ax.scatter(d.mean_predicted, d.observed_win_rate, s=40 + 16 * np.sqrt(d.candidate_weeks), marker=markers[source],
                           color=mcolors[source], alpha=.8, edgecolor="white", linewidth=.6)
                source_handles.append(ax.scatter([], [], s=60, marker=markers[source], color=mcolors[source], alpha=.8, label=source))
            ax.set_xlim(-.03, 1.03)
            ax.set_ylim(-.05, 1.05)
            ax.set_xlabel("Mean predicted probability in bin", fontsize=12)
            ax.set_ylabel("Observed win rate", fontsize=12)
            ax.set_title(f"{c}h cutoff: " + ("all 90 weeks (market only)" if j == 0 else "test weeks (18)"), fontsize=12.5)
            ax.tick_params(labelsize=11.5)
            source_legend = ax.legend(handles=source_handles, loc="upper left", fontsize=11, frameon=False)
            ax.add_artist(source_legend)
            ref = (10, 100, 700) if j == 0 else (2, 20, 200)
            size_handles = [ax.scatter([], [], s=40 + 16 * np.sqrt(n_), color="#b8c0ca", edgecolor="white", label=f"{n_} rows") for n_ in ref]
            ax.legend(handles=size_handles, title="Area = rows in bin", loc="lower right", fontsize=11, title_fontsize=11,
                      frameon=False, labelspacing=1.1)
            height = .09 * (len(sources) + 1)
            names = [{"Market price": "Market", LR_METHODS[2]: "LR-C", RF_METHOD: "RF"}[s] for s in sources]
            table = ax.table(cellText=[[nm_] + row for nm_, row in zip(names, cells)], colLabels=["Source"] + edges,
                             colWidths=[.2] + [.16] * 5, loc="bottom", bbox=[0, -.32 - height, 1, height])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            ax.text(0, -.235, "Candidate-weeks per probability bin", transform=ax.transAxes, fontsize=11, color="#555", va="top")
        fig.text(.5, .955, "Diagnostic only. One point per candidate-week; candidates in the same week are correlated; "
                 "few independent weeks (90 all, 18 test).", fontsize=11, color="#555", ha="center")
        fig.savefig(out / f"04_calibration_{c}h.png", bbox_inches="tight")
        plt.close(fig)

    ct = common_table[common_table.split == "test"]
    ct = ct[ct["sample"] == "full sample"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    spec = [("top1_accuracy", "Top-1 accuracy (%)", 100), ("log_loss_week_equal", "Binary log loss (lower = better)", 1),
            ("brier_week_equal", "Brier score (lower = better)", 1)]
    for ax, (col, title, scale) in zip(axes, spec):
        for k, c in enumerate(CUTOFFS):
            d = ct[ct.cutoff_hours == c].set_index("method").loc[METHODS]
            xs = np.arange(5) + (k - .5) * .38
            ax.bar(xs, d[col] * scale, width=.38, color=["#486581", "#d9480f"][k], label=f"{c}h before close")
            for x, row in zip(xs, d.itertuples()):
                text = f"{row.top1_correct}/{row.weeks}" if col == "top1_accuracy" else f"{getattr(row, col):.3f}"
                ax.text(x, getattr(row, col) * scale, text, ha="center", va="bottom", fontsize=10)
        ax.set(xticks=range(5), xticklabels=[short[x] for x in METHODS], title=title)
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.title.set_fontsize(13)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, frameon=False, fontsize=12)
    fig.text(.01, .005, "Test weeks (n=18). The eventual winner was listed at both cutoffs in all 18 test weeks, so the full-sample and "
             "common-week test results are identical.", fontsize=10.5, color="#555")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / "05_cutoff_comparison.png", bbox_inches="tight")
    plt.close(fig)

    g = r24["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for method, size in zip(LR_METHODS, [260, 170, 90]):
        d = g[g.model == method]
        axes[0].plot(d.param_C, d.val_log_loss, marker="o", label=method, color=colors[method])
        s = d[d.selected]
        axes[0].scatter(s.param_C, s.val_log_loss, s=size, facecolors="none", edgecolors=colors[method], linewidths=1.8, zorder=5)
    axes[0].set(xscale="log", xlabel="Inverse regularization strength C", ylabel="Validation log loss (week-equal)",
                title="Logistic regression tuning (ring = selected)")
    axes[0].legend(fontsize=11)
    d = g[g.model == RF_METHOD]
    pivot = d.pivot(index="param_max_depth", columns="param_min_samples_leaf", values="val_log_loss")
    im = axes[1].imshow(pivot.to_numpy(), cmap="viridis_r", aspect="auto")
    axes[1].set(xticks=range(pivot.shape[1]), xticklabels=[int(x) for x in pivot.columns],
                yticks=range(pivot.shape[0]), yticklabels=[int(x) for x in pivot.index],
                xlabel="min_samples_leaf", ylabel="max_depth", title="Random Forest validation log loss (* = selected)")
    for (i, j), v in np.ndenumerate(pivot.to_numpy()):
        sel = bool(d[(d.param_max_depth == pivot.index[i]) & (d.param_min_samples_leaf == pivot.columns[j])].selected.iloc[0])
        axes[1].text(j, i, f"{v:.4f}" + ("*" if sel else ""), ha="center", va="center", color="white", fontsize=11.5)
    for ax_ in axes:
        ax_.title.set_fontsize(12.5)
        ax_.xaxis.label.set_fontsize(12)
        ax_.yaxis.label.set_fontsize(12)
        ax_.tick_params(labelsize=11.5)
    fig.colorbar(im, ax=axes[1], fraction=.046)
    fig.tight_layout()
    fig.savefig(out / "06_validation_tuning_24h.png", bbox_inches="tight")
    plt.close(fig)

    inc = r24["audit"].query("status == 'included'")
    fig, ax = plt.subplots(figsize=(13, 3.2))
    palette = {"train": "#486581", "validation": "#e9a23b", "test": "#d9480f"}
    for split in ["train", "validation", "test"]:
        d = inc[inc.split == split]
        a, b = pd.to_datetime(d.prediction_time_utc.min()), pd.to_datetime(d.prediction_time_utc.max())
        ax.barh(0, (b - a).days + 7, left=a.tz_localize(None), height=.5, color=palette[split])
        ax.text(a.tz_localize(None) + (b - a) / 2, 0, f"{split}\n{len(d)} weeks\n{a:%Y-%m-%d} to {b:%Y-%m-%d}",
                ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    ax.set(yticks=[], xlabel="24h prediction time (UTC); whole weeks, chronological; the 72h cutoff reuses the same week assignment",
           title="Temporal split of the 90 included weeks")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "07_temporal_split_timeline.png", bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------------------------- main
def run(data_dir, output_dir, figure_dir):
    t0 = time.time()
    data_dir, out, fig_dir = Path(data_dir), Path(output_dir), Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    markets, candles, source = load_inputs(data_dir)
    res, split_map = {}, None
    for cutoff in CUTOFFS:
        res[cutoff] = run_cutoff(markets, candles, cutoff, split_map)
        split_map = res[cutoff]["split_map"]
    common_table, common_weeks = common_week_table(res)
    cal = []
    for c in CUTOFFS:
        d, test = res[c]["data"], res[c]["data"][res[c]["data"].split == "test"]
        cal.append(calibration_table(d, "price_now", "Market price", c, "all included weeks (market only)"))
        cal.append(calibration_table(test, "price_now", "Market price", c, "test weeks"))
        for method in [LR_METHODS[2], RF_METHOD]:
            cal.append(calibration_table(test[test.price_now.notna()], method, method, c, "test weeks"))
    res["calibration"] = pd.concat(cal, ignore_index=True)
    stack = lambda key: pd.concat([res[c][key] for c in CUTOFFS], ignore_index=True)
    save_csv(stack("features"), out / "features_by_cutoff.csv")
    save_csv(stack("audit").assign(cutoff_hours=lambda d: np.repeat(CUTOFFS, [len(res[c]["audit"]) for c in CUTOFFS])), out / "weekly_audit_by_cutoff.csv")
    save_csv(stack("grid"), out / "tuning_all_results.csv")
    save_csv(pd.DataFrame([dict(cutoff_hours=c, model=m, **{k: v for k, v in r.items() if k not in ("model", "cutoff_hours")})
                           for c in CUTOFFS for m, r in res[c]["chosen"].items()]).assign(tie_key=lambda d: d.tie_key.astype(str)),
             out / "tuning_selected_parameters.csv")
    save_csv(stack("data").drop(columns=[]), out / "candidate_predictions.csv")
    save_csv(stack("predictions"), out / "weekly_selections.csv")
    save_csv(stack("metrics"), out / "metrics.csv")
    save_csv(stack("paired"), out / "paired_comparisons.csv")
    save_csv(stack("coefficients"), out / "logistic_coefficients.csv")
    save_csv(stack("importances"), out / "forest_importances.csv")
    save_csv(common_table, out / "cutoff_common_week_comparison.csv")
    save_csv(res["calibration"], out / "calibration_bins.csv")
    save_csv(describe(res), out / "data_audit_summary.csv")
    save_csv(market_all_weeks_table(res, common_weeks), out / "market_leader_all_weeks.csv")
    figures(res, common_table, common_weeks, fig_dir)
    protocol = Path(__file__).resolve().parent.parent / "docs" / "PROTOCOL_v2.md"
    record = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": round(time.time() - t0, 1),
        "seed": SEED, "cutoffs_hours": CUTOFFS, "candles_source": source,
        "input_sha256": {p.name: sha256(p) for p in sorted(data_dir.glob("*")) if p.suffix in (".csv", ".parquet")},
        "script_sha256": sha256(__file__), "protocol_sha256": sha256(protocol) if protocol.exists() else None,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "lr_grid_C": LR_GRID, "rf_grid_(max_depth,min_samples_leaf)": RF_GRID, "rf_trees": RF_TREES,
        "selection": "validation week-equal binary log loss; ties < 1e-6 -> more regularized",
        "clip_eps": EPS, "common_weeks": len(common_weeks),
        "selected_parameters": {str(c): {m: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()
                                              if k in ("param_C", "param_max_depth", "param_min_samples_leaf", "val_log_loss")}
                                        for m, r in res[c]["chosen"].items()} for c in CUTOFFS},
        "test_set_note": "Test weeks were viewed in v1; this is an exploratory re-analysis on existing data."}
    (out / "run_record.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(stack("metrics")[lambda d: d.split.isin(["validation", "test"])].round(4).to_string(index=False))
    print(pd.read_csv(out / "tuning_selected_parameters.csv").to_string(index=False))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--data-dir", type=Path, default=root / "input_data")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--figure-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.figure_dir)
