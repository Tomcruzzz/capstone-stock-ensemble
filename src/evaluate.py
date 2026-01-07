# evaluate.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DEFAULT_MARKET,
    DEFAULT_PEERS,
    DEFAULT_SECTOR_ETF,
    DEFAULT_TARGET,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from .features import feature_groups, make_feature_frame
from .models import build_base_models, build_meta_model
from .sector import compute_breadth, get_sector_series
from .stack import make_meta_matrix, oof_predictions
from .utils import compute_returns, load_ticker_csv, mae, rmse, save_json, sign_accuracy, to_date_index


def _decision_labels(r_hat, theta):
    labels = []
    for r in r_hat:
        if r >= theta:
            labels.append("Buy")
        elif r <= -theta:
            labels.append("Sell")
        else:
            labels.append("Hold")
    return labels


def _tune_threshold(y_true, y_pred):
    grid = np.linspace(0.002, 0.02, 10)
    best = (None, -1, -1)
    for theta in grid:
        labels = _decision_labels(y_pred, theta)
        # Map labels to sign for rough alignment
        pred_sign = np.array([1 if x == "Buy" else -1 if x == "Sell" else 0 for x in labels])
        true_sign = np.sign(y_true)
        score = np.mean(pred_sign == true_sign)
        # Prefer larger theta when score ties to keep neutral zone wide.
        if score > best[1] or (score == best[1] and theta > best[0]):
            best = (theta, score, len(labels))
    return best[0]


def _metrics_table(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": round(mae(y_true, y_pred), 6),
        "RMSE": round(rmse(y_true, y_pred), 6),
        "SignAcc": round(sign_accuracy(y_true, y_pred), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate next-day return models.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--sector-etf", default=DEFAULT_SECTOR_ETF)
    parser.add_argument("--peers", default=",".join(DEFAULT_PEERS))
    parser.add_argument("--k-peers", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_df = compute_returns(load_ticker_csv(data_dir, args.target))
    market_df = compute_returns(load_ticker_csv(data_dir, args.market))
    target_df = to_date_index(target_df)
    market_df = to_date_index(market_df)

    peer_candidates = [p.strip().upper() for p in args.peers.split(",") if p.strip()]
    sector_df, peers, sector_etf = get_sector_series(
        data_dir,
        args.target,
        args.sector_etf,
        peer_candidates,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        k=args.k_peers,
    )
    breadth_peers = peers if peers else [p for p in peer_candidates if p.upper() != args.target.upper()]
    breadth_df = compute_breadth(data_dir, breadth_peers)

    feature_df = make_feature_frame(target_df, market_df[["return"]], sector_df[["sec_return"]], breadth_df)

    groups = feature_groups([c for c in feature_df.columns if c not in ["target", "adj_close"]])
    own_cols = groups["own"] + groups["calendar"]
    own_market_cols = own_cols + groups["market"]
    full_cols = own_market_cols + groups["sector"]

    # Split by date
    train_mask = (feature_df.index >= TRAIN_START) & (feature_df.index <= TRAIN_END)
    val_mask = (feature_df.index >= VAL_START) & (feature_df.index <= VAL_END)
    test_mask = (feature_df.index >= TEST_START) & (feature_df.index <= TEST_END)

    train = feature_df.loc[train_mask]
    val = feature_df.loc[val_mask]
    test = feature_df.loc[test_mask]

    X_train = train[full_cols].to_numpy()
    y_train = train["target"].to_numpy()
    X_val = val[full_cols].to_numpy()
    y_val = val["target"].to_numpy()
    X_test = test[full_cols].to_numpy()
    y_test = test["target"].to_numpy()

    base_models = build_base_models()

    # OOF predictions on validation
    oof = oof_predictions(base_models, X_train, y_train, X_val, y_val, n_splits=4)
    oof["target"] = y_val

    Z_val, base_order = make_meta_matrix(oof)
    meta_model = build_meta_model()
    meta_model.fit(Z_val, y_val)
    ens_val_pred = meta_model.predict(Z_val)

    theta = _tune_threshold(y_val, ens_val_pred)

    # Refit base models on train+val and predict test
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    base_test_preds = {}
    for name, model in base_models.items():
        model.fit(X_train_full, y_train_full)
        base_test_preds[name] = model.predict(X_test)

    Z_test = np.column_stack([base_test_preds[name] for name in base_order])
    ens_test_pred = meta_model.predict(Z_test)

    # Metrics table
    metrics = []
    metrics.append(_metrics_table("Naive (0)", y_test, np.zeros_like(y_test)))
    for name in base_order:
        metrics.append(_metrics_table(f"{name} (+mkt+sec)", y_test, base_test_preds[name]))
    metrics.append(_metrics_table("Ensemble (stack)", y_test, ens_test_pred))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    # Ablations (ElasticNet)
    ablations = []
    for feat_name, cols in [
        ("own_only", own_cols),
        ("own+market", own_market_cols),
        ("own+market+sector", full_cols),
    ]:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42)),
            ]
        )
        model.fit(feature_df.loc[train_mask | val_mask, cols].to_numpy(),
                  feature_df.loc[train_mask | val_mask, "target"].to_numpy())
        preds = model.predict(feature_df.loc[test_mask, cols].to_numpy())
        ablations.append(
            {
                "feature_set": feat_name,
                "model": "ElasticNet",
                "MAE": round(mae(y_test, preds), 6),
                "RMSE": round(rmse(y_test, preds), 6),
                "SignAcc": round(sign_accuracy(y_test, preds), 6),
            }
        )
    pd.DataFrame(ablations).to_csv(output_dir / "ablations.csv", index=False)

    # Predictions CSV
    decisions = _decision_labels(ens_test_pred, theta)
    notes = []
    for idx, row in test.iterrows():
        mkt = row.get("mkt_return", 0.0)
        sec = row.get("sec_return", 0.0)
        note = f"market {'up' if mkt >= 0 else 'down'}, sector {'up' if sec >= 0 else 'down'}"
        notes.append(note)

    pred_df = pd.DataFrame(
        {
            "date": test.index.strftime("%Y-%m-%d"),
            "ticker": args.target.upper(),
            "r_hat_next": ens_test_pred,
            "p_hat_next": test["adj_close"].to_numpy() * (1.0 + ens_test_pred),
            "decision": decisions,
            "notes": notes,
        }
    )
    pred_df.to_csv(output_dir / "predictions_test.csv", index=False)

    # Save models and metadata
    joblib.dump(
        {
            "base_models": base_models,
            "meta_model": meta_model,
            "base_order": base_order,
        },
        output_dir / "models.joblib",
    )

    save_json(
        output_dir / "metadata.json",
        {
            "target": args.target.upper(),
            "market": args.market.upper(),
            "sector_etf": sector_etf,
            "peers": peers,
            "threshold": theta,
            "feature_cols": {
                "own": own_cols,
                "own_market": own_market_cols,
                "full": full_cols,
            },
            "date_splits": {
                "train": [TRAIN_START, TRAIN_END],
                "val": [VAL_START, VAL_END],
                "test": [TEST_START, TEST_END],
            },
        },
    )

    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()

