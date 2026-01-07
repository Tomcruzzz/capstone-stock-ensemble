# predict.py
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .config import DEFAULT_MARKET, DEFAULT_PEERS, DEFAULT_SECTOR_ETF, DEFAULT_TARGET
from .features import make_feature_frame
from .sector import compute_breadth, get_sector_series, sector_return_from_peers
from .utils import compute_returns, load_json, load_ticker_csv, to_date_index


def main():
    parser = argparse.ArgumentParser(description="Predict next-day return for a date.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--ticker", default=DEFAULT_TARGET)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--sector-etf", default=DEFAULT_SECTOR_ETF)
    parser.add_argument("--peers", default=",".join(DEFAULT_PEERS))
    parser.add_argument("--k-peers", type=int, default=10)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    metadata = load_json(artifacts / "metadata.json")

    data_dir = Path(args.data_dir)
    target_df = compute_returns(load_ticker_csv(data_dir, args.ticker))
    market_df = compute_returns(load_ticker_csv(data_dir, args.market))
    target_df = to_date_index(target_df)
    market_df = to_date_index(market_df)

    peer_candidates = [p.strip().upper() for p in args.peers.split(",") if p.strip()]
    fixed_peers = metadata.get("peers", [])
    sector_etf = metadata.get("sector_etf")
    if sector_etf:
        sector_df, _, _ = get_sector_series(
            data_dir,
            args.ticker,
            sector_etf,
            peer_candidates,
            train_start=metadata["date_splits"]["train"][0],
            train_end=metadata["date_splits"]["train"][1],
            k=args.k_peers,
        )
        peers = fixed_peers
    else:
        peers = fixed_peers if fixed_peers else peer_candidates
        sector_df = sector_return_from_peers(data_dir, peers).rename(columns={"return": "sec_return"})

    breadth_peers = peers if peers else [p for p in peer_candidates if p.upper() != args.ticker.upper()]
    breadth_df = compute_breadth(data_dir, breadth_peers)

    feature_df = make_feature_frame(
        target_df, market_df[["return"]], sector_df[["sec_return"]], breadth_df, drop_target_na=False
    )
    date = np.datetime64(args.date)
    if date not in feature_df.index:
        raise ValueError(f"date not found in features: {args.date}")

    row = feature_df.loc[date]
    feature_cols = metadata["feature_cols"]["full"]
    # Check for missing feature columns
    missing_cols = [col for col in feature_cols if col not in row.index]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    X = row[feature_cols].to_numpy().reshape(1, -1)

    models = joblib.load(artifacts / "models.joblib")
    base_order = models["base_order"]
    base_models = models["base_models"]
    meta_model = models["meta_model"]

    base_preds = []
    for name in base_order:
        base_preds.append(base_models[name].predict(X)[0])
    Z = np.array(base_preds).reshape(1, -1)
    r_hat = float(meta_model.predict(Z)[0])
    p_hat = float(row["adj_close"] * (1.0 + r_hat))

    theta = metadata["threshold"]
    if r_hat >= theta:
        decision = "Buy"
    elif r_hat <= -theta:
        decision = "Sell"
    else:
        decision = "Hold"

    drivers = {
        "own": {
            "r_t": float(row.get("own_lag_0", 0.0)),
            "mom20": float(row.get("own_roll_mean_20", 0.0)),
            "vol20": float(row.get("own_roll_vol_20", 0.0)),
        },
        "market": {
            "r_mkt_t": float(row.get("mkt_return", 0.0)),
            "breadth_t": float(row.get("breadth", 0.0)),
        },
        "sector": {
            "r_sec_t": float(row.get("sec_return", 0.0)),
            "beta_mkt": float(row.get("factor_beta_mkt", 0.0)),
            "beta_sec": float(row.get("factor_beta_sec", 0.0)),
            "idio_resid_t": float(row.get("factor_resid", 0.0)),
        },
    }

    payload = {
        "date": args.date,
        "ticker": args.ticker.upper(),
        "r_hat_next": r_hat,
        "p_hat_next": p_hat,
        "decision": decision,
        "drivers": drivers,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

