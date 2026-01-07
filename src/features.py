# features.py
import numpy as np
import pandas as pd

from .utils import safe_log, to_date_index


def _rolling_zscore(series, window):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def _add_return_lags(df, col, prefix, max_lag=5):
    out = {}
    for i in range(max_lag + 1):
        name = f"{prefix}lag_{i}"
        out[name] = df[col].shift(i)
    return out


def _add_rolling_stats(df, col, prefix, windows):
    out = {}
    for w in windows:
        out[f"{prefix}roll_mean_{w}"] = df[col].rolling(w).mean()
        out[f"{prefix}roll_vol_{w}"] = df[col].rolling(w).std()
    return out


def _compute_expanding_betas(y, x1, x2, min_obs=30):
    n = len(y)
    betas = np.full((n, 3), np.nan)
    resid = np.full(n, np.nan)

    sum1 = sum2 = sum11 = sum22 = sum12 = 0.0
    sumy = sum1y = sum2y = 0.0
    count = 0

    for i in range(n):
        yi = y[i]
        x1i = x1[i]
        x2i = x2[i]
        if np.isnan(yi) or np.isnan(x1i) or np.isnan(x2i):
            betas[i] = np.array([np.nan, np.nan, np.nan])
            resid[i] = np.nan
            continue
        count += 1
        sum1 += x1i
        sum2 += x2i
        sum11 += x1i * x1i
        sum22 += x2i * x2i
        sum12 += x1i * x2i
        sumy += yi
        sum1y += x1i * yi
        sum2y += x2i * yi

        if count < min_obs:
            betas[i] = np.array([np.nan, np.nan, np.nan])
            resid[i] = np.nan
            continue

        xtx = np.array(
            [
                [count, sum1, sum2],
                [sum1, sum11, sum12],
                [sum2, sum12, sum22],
            ]
        )
        xty = np.array([sumy, sum1y, sum2y])
        try:
            coeffs = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            coeffs = np.array([np.nan, np.nan, np.nan])
        betas[i] = coeffs
        resid[i] = yi - (coeffs[0] + coeffs[1] * x1i + coeffs[2] * x2i)

    return betas, resid


def make_feature_frame(target_df, market_ret, sector_ret, breadth, min_history_days=70, drop_target_na=True):
    target = to_date_index(target_df)
    market_ret = market_ret.rename(columns={market_ret.columns[0]: "mkt_return"})
    sector_ret = sector_ret.rename(columns={sector_ret.columns[0]: "sec_return"})
    breadth = breadth.rename(columns={breadth.columns[0]: "breadth"})

    base = target[["Adj Close", "High", "Low", "Close", "Volume", "return"]].copy()
    base = base.join(market_ret, how="left").join(sector_ret, how="left").join(breadth, how="left")

    features = {}

    # Own stock features
    features.update(_add_return_lags(base, "return", "own_", max_lag=5))
    features.update(_add_rolling_stats(base, "return", "own_", [5, 20, 63]))

    hlc = (base["High"] - base["Low"]) / base["Close"]
    features["own_hlc"] = hlc
    features["own_hlc_z20"] = _rolling_zscore(hlc, 20)
    features["own_hlc_z63"] = _rolling_zscore(hlc, 63)

    adj = base["Adj Close"]
    for w in [5, 20, 63]:
        sma = adj.rolling(w).mean()
        features[f"own_sma_gap_{w}"] = (adj / sma) - 1.0

    log_vol = safe_log(base["Volume"])
    features["own_log_vol"] = log_vol
    features["own_log_vol_z21"] = _rolling_zscore(log_vol, 21)
    features["own_log_vol_delta_5"] = log_vol - log_vol.shift(5)
    features["own_log_vol_delta_20"] = log_vol - log_vol.shift(20)

    # Market features
    features["mkt_return"] = base["mkt_return"]
    features.update(_add_return_lags(base, "mkt_return", "mkt_", max_lag=5))
    features.update(_add_rolling_stats(base, "mkt_return", "mkt_", [20, 63]))

    # Sector features
    features["sec_return"] = base["sec_return"]
    features.update(_add_return_lags(base, "sec_return", "sec_", max_lag=5))
    features.update(_add_rolling_stats(base, "sec_return", "sec_", [20, 63]))

    # Breadth
    features["breadth"] = base["breadth"]
    for i in range(1, 6):
        features[f"breadth_lag_{i}"] = base["breadth"].shift(i)

    # Factor betas and residuals
    y = base["return"].to_numpy()
    x1 = base["mkt_return"].to_numpy()
    x2 = base["sec_return"].to_numpy()
    betas, resid = _compute_expanding_betas(y, x1, x2, min_obs=30)
    features["factor_alpha"] = betas[:, 0]
    features["factor_beta_mkt"] = betas[:, 1]
    features["factor_beta_sec"] = betas[:, 2]
    features["factor_resid"] = resid

    # Calendar
    features["cal_dow"] = base.index.dayofweek
    features["cal_month"] = base.index.month

    feature_df = pd.DataFrame(features, index=base.index)
    feature_df = pd.get_dummies(feature_df, columns=["cal_dow", "cal_month"], drop_first=False)

    feature_df["target"] = base["return"].shift(-1)
    feature_df["adj_close"] = base["Adj Close"]

    # Drop early rows with insufficient history
    if min_history_days:
        feature_df = feature_df.iloc[min_history_days:]

    feature_cols = [c for c in feature_df.columns if c != "target"]
    feature_df = feature_df.dropna(subset=feature_cols)
    if drop_target_na:
        feature_df = feature_df.dropna(subset=["target"])
    return feature_df


def feature_groups(columns):
    own = [c for c in columns if c.startswith("own_")]
    mkt = [c for c in columns if c.startswith("mkt_") or c == "mkt_return"]
    sec = [c for c in columns if c.startswith("sec_") or c == "sec_return"]
    breadth = [c for c in columns if c.startswith("breadth")]
    factor = [c for c in columns if c.startswith("factor_")]
    cal = [c for c in columns if c.startswith("cal_")]
    return {
        "own": own,
        "market": mkt + breadth,
        "sector": sec + factor,
        "calendar": cal,
    }

