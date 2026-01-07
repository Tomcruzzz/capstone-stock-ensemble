# utils.py
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def find_ticker_path(data_dir, ticker):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    candidates = [p for p in data_dir.glob("*.csv") if p.stem.upper() == ticker.upper()]
    if candidates:
        return candidates[0]
    # Fallback: case-insensitive search
    for p in data_dir.glob("*.csv"):
        if p.stem.lower() == ticker.lower():
            return p
    return None


def load_ticker_csv(data_dir, ticker):
    path = find_ticker_path(data_dir, ticker)
    if path is None:
        raise FileNotFoundError(f"ticker csv not found for {ticker} in {data_dir}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_returns(df):
    out = df.copy()
    out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")
    out["return"] = out["Adj Close"].pct_change()
    return out


def to_date_index(df):
    df = df.copy()
    df = df.set_index("Date").sort_index()
    return df


def date_mask(df, start, end):
    return (df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))


def safe_log(series):
    return np.log(series.replace(0, np.nan))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def sign_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="ascii") as f:
        return json.load(f)

