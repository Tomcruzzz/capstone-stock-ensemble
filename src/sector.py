# sector.py
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import compute_returns, load_ticker_csv, to_date_index


def available_tickers(data_dir):
    data_dir = Path(data_dir)
    return sorted([p.stem.upper() for p in data_dir.glob("*.csv")])


def build_peer_sector(data_dir, target_ticker, peer_candidates, train_start, train_end, k=10):
    target_df = compute_returns(load_ticker_csv(data_dir, target_ticker))
    target_df = to_date_index(target_df)
    target_train = target_df.loc[train_start:train_end]["return"].dropna()

    peers = []
    scores = []
    for ticker in peer_candidates:
        if ticker.upper() == target_ticker.upper():
            continue
        try:
            peer_df = compute_returns(load_ticker_csv(data_dir, ticker))
        except FileNotFoundError:
            continue
        peer_df = to_date_index(peer_df)
        aligned = pd.concat([target_train, peer_df["return"]], axis=1, join="inner").dropna()
        if aligned.shape[0] < 30:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.isna(corr):
            continue
        peers.append(ticker)
        scores.append(corr)

    if not peers:
        raise ValueError("no peers found for sector construction")

    order = np.argsort(scores)[::-1]
    top_peers = [peers[i] for i in order[:k]]
    return top_peers


def sector_return_from_peers(data_dir, peers):
    returns = []
    for ticker in peers:
        df = compute_returns(load_ticker_csv(data_dir, ticker))
        df = to_date_index(df)
        returns.append(df["return"].rename(ticker))
    combined = pd.concat(returns, axis=1, join="outer")
    sector_ret = combined.mean(axis=1, skipna=True)
    return sector_ret.to_frame("return")


def get_sector_series(data_dir, target_ticker, sector_etf, peer_candidates, train_start, train_end, k=10):
    if sector_etf:
        try:
            etf_df = compute_returns(load_ticker_csv(data_dir, sector_etf))
        except FileNotFoundError:
            sector_etf = None
        else:
            etf_df = to_date_index(etf_df)
            return etf_df[["return"]].rename(columns={"return": "sec_return"}), [], sector_etf

    peers = build_peer_sector(
        data_dir=data_dir,
        target_ticker=target_ticker,
        peer_candidates=peer_candidates,
        train_start=train_start,
        train_end=train_end,
        k=k,
    )
    sec_df = sector_return_from_peers(data_dir, peers)
    sec_df = sec_df.rename(columns={"return": "sec_return"})
    return sec_df, peers, None


def compute_breadth(data_dir, peers):
    if not peers:
        raise ValueError("breadth peers list is empty")
    returns = []
    for ticker in peers:
        df = compute_returns(load_ticker_csv(data_dir, ticker))
        df = to_date_index(df)
        returns.append(df["return"].rename(ticker))
    combined = pd.concat(returns, axis=1, join="outer")
    breadth = (combined > 0).sum(axis=1) / combined.shape[1]
    return breadth.to_frame("breadth")

