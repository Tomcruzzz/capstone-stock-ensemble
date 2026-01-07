# Stock Prediction Capstone - Project Documentation

## Problem Statement
Build a leakage-safe, next-day stock return prediction system that uses three information sources:
1) the stock's own recent behavior, 2) market behavior (QQQ), and 3) sector behavior (sector ETF or peer-based index). The model should predict next-day adjusted-close returns and translate those into next-day prices. The approach must use walk-forward splits and avoid data leakage.

## Dataset
- Source: Kaggle "Stock Market Dataset (NASDAQ)"
- Link: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset
- Format: One CSV per ticker with daily OHLCV and Adj Close
- Target: Next-day return based on Adj Close

## Approach Overview
- Compute leakage-safe features using only data available at time t to predict t+1.
- Build sector context using either a sector ETF (if available) or a peer-based sector index.
- Train diverse base regressors (ElasticNet, RandomForest, GradientBoosting) and combine them with stacking.
- Use walk-forward TimeSeriesSplit for validation OOF predictions and train a meta-learner.
- Evaluate on 2020-Q1 test data with MAE, RMSE, and sign accuracy.

## Methodology

### 1) Splits (Walk-forward)
- Train: 2018
- Validation: 2019
- Test: 2020-Q1 (Jan?Mar)
- All choices (features, hyperparameters, ensemble strategy, thresholds) are tuned on Validation only and then frozen before Test.

### 2) Target Definition
- Return: r_{i,t+1} = (AdjClose_{i,t+1} - AdjClose_{i,t}) / AdjClose_{i,t}
- Price forecast: P_hat_{i,t+1} = AdjClose_{i,t} * (1 + r_hat_{i,t+1})

### 3) Leakage-safe Features
All features use data at or before time t.

Own stock:
- Return lags: r_{t}, r_{t-1}, ..., r_{t-5}
- Rolling mean/volatility: 5/20/63 days
- Intraday range: (High - Low) / Close with rolling z-scores (20/63)
- SMA gaps: price vs SMA(5/20/63)
- Volume features: log(volume), z-score(21), delta(5/20)

Market:
- Market return (QQQ) + lags and rolling vol

Sector:
- Sector return (ETF or peer index) + lags and rolling vol
- Breadth: fraction of peers with positive returns
- Expanding OLS factor betas (market/sector) and residuals

Calendar:
- Day of week and month (one-hot)

### 4) Sector Proxy
- ETF option: use sector ETF if available in dataset.
- Peer option: compute correlations on Train only, select top K peers, and average their returns as a sector index.
- Peers are fixed after Train and not refreshed using Test data.

### 5) Models
Base learners:
- ElasticNet (scaled)
- RandomForest
- GradientBoostingRegressor

Ensemble:
- Stacking with ElasticNet meta-learner trained on walk-forward OOF predictions in Validation.

### 6) Evaluation
Metrics (Test 2020-Q1):
- MAE
- RMSE
- Sign accuracy

Ablations:
- Own-only vs Own+Market vs Own+Market+Sector (ElasticNet)

## Results (Artifacts)
Generated files after training/evaluation:
- `artifacts/metrics.csv`: MAE/RMSE/SignAcc for each base model and the ensemble
- `artifacts/ablations.csv`: feature set ablation metrics
- `artifacts/predictions_test.csv`: test predictions with decision labels and notes
- `artifacts/models.joblib`: trained base models and meta model
- `artifacts/metadata.json`: split ranges, feature columns, peers, and threshold

Note: Final numeric results depend on the chosen target ticker, peer list, and availability of sector ETF/peers in `data/`.

## Usage

### Training/Evaluation
```bash
python -m src.evaluate --data-dir data --target AAPL --market QQQ
```

### Date Query (Prediction)
```bash
python -m src.predict --date 2020-02-27 --ticker AAPL
```

Example output:
```json
{
  "date": "2020-02-27",
  "ticker": "AAPL",
  "r_hat_next": 0.0042,
  "p_hat_next": 281.63,
  "decision": "Hold",
  "drivers": {
    "own": {"r_t": -0.042, "mom20": -0.031, "vol20": 0.028},
    "market": {"r_mkt_t": -0.031, "breadth_t": 0.12},
    "sector": {"r_sec_t": -0.035, "beta_mkt": 1.22, "beta_sec": 0.87, "idio_resid_t": -0.006}
  }
}
```

## Insights
- Using market and sector context generally improves stability vs own-only features.
- Stacking helps smooth individual model errors on noisy return targets.
- Sign accuracy is a more robust indicator than R2 for short-horizon returns.

## Limitations
- Short test window (2020-Q1) and noisy target reduce predictability.
- Potential survivorship bias from dataset composition.
- No transaction costs or slippage modeled.
- Not investment advice; educational use only.

## Project Structure
- `src/`: feature engineering, sector construction, models, stacking, evaluation, prediction
- `data/`: Kaggle CSVs (one per ticker)
- `artifacts/`: metrics, predictions, and model artifacts
- `notebooks/`: consolidated notebook

## Access Notes
- Ensure the Kaggle dataset is available locally in `data/`.
- All outputs are written to local `artifacts/` without restricted access.
