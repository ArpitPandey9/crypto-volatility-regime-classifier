# Crypto Volatility Regime Classifier

Experimental Python research project for **classifying contemporaneous crypto volatility regimes** with engineered market features, expanding-window splits, XGBoost, and optional SHAP analysis.

## Current implementation

The public source currently includes:

- daily market-data retrieval for BTC, ETH, LTC, BNB, and SOL through `yfinance`
- rolling-return, volatility, momentum, RSI, MACD, Bollinger-width, and moving-average features
- BTC 7-day rolling-volatility regimes defined from 33% / 66% quantile thresholds
- an expanding-window train/test split helper
- XGBoost training and per-fold classification reporting
- a reusable SHAP summary-plot helper that requires the caller to provide the fitted model and exact feature matrix

## Important interpretation

The current target is a **same-period BTC volatility-regime classification target** derived from BTC 7-day rolling volatility. This repository does **not** establish forward predictive power, trading alpha, or production forecasting performance.

Reported fold accuracy from an ad-hoc run should not be interpreted as an out-of-sample investment result. A research-grade forecasting study would require stricter target timing, leakage controls, benchmark comparisons, reproducible frozen data, and independently validated evaluation metrics.

## Environment

Verified with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validation

```bash
python -m compileall -q src tests
python -m pytest -q
```

The repository CI also validates that the tracked notebook is valid JSON.

## Run the classifier experiment

```bash
python -m src.model
```

This command downloads market data at runtime, so results depend on provider availability and the observation window returned at execution time.

## Explainability helper

`src/shap_analysis.py` exposes `plot_shap_summary(model, features, output_path=None)`. It intentionally does not guess or reconstruct the feature matrix.

## Example artifacts

![SHAP summary plot](SHAP%20Summary%20Plot.png)

![XGBoost feature importance](XGBoost%20Feature%20Importance%20Bar%20Chart.png)

These images are retained as example research artifacts; they are not presented as independently reproduced performance evidence.

## Limitations

This is a research prototype, not an investment recommendation, production risk model, or representation of any employer or client.

## License

See `LICENSE`.
