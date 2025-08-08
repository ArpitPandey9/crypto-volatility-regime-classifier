import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from features import prepare_features

def walk_forward_cv(X, y, initial_train_size, step_size):
    n_samples = len(X)
    splits = []
    start = initial_train_size
    while start + step_size <= n_samples:
        train_idx = list(range(start))
        test_idx = list(range(start, start + step_size))
        splits.append((train_idx, test_idx))
        start += step_size
    return splits

def main():
    tickers = ['BTC-USD', 'ETH-USD', 'LTC-USD', 'BNB-USD', 'SOL-USD']
    dfs = {}
    for t in tickers:
        dfs[t] = yf.download(t, period='3y', interval='1d', auto_adjust=True)

    feature_dfs = [prepare_features(dfs[t], t.split('-')[0].lower()) for t in tickers]
    data = pd.concat(feature_dfs, axis=1).dropna()

    vol_7d = data['btc_volatility_7d']
    if isinstance(vol_7d, pd.DataFrame):
        vol_7d = vol_7d.iloc[:, 0]

    low_val = vol_7d.quantile(0.33)
    high_val = vol_7d.quantile(0.66)
    bins = [-np.inf, low_val, high_val, np.inf]
    labels = ['Low', 'Medium', 'High']

    data['volatility_regime'] = pd.cut(vol_7d, bins=bins, labels=labels)
    data.dropna(inplace=True)

    X = data.drop(columns=['volatility_regime'])
    y = data['volatility_regime']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    initial_train = int(len(X) * 0.6)
    step = int(len(X) * 0.1)
    splits = walk_forward_cv(X, y_enc, initial_train, step)

    accuracies = []
    fold = 1
    best_acc = 0
    best_model = None

    for train_idx, test_idx in splits:
        print(f"Fold {fold}: Train {len(train_idx)} samples, Test {len(test_idx)} samples")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = model

        fold += 1

    print(f"\\nAverage Walk-Forward Accuracy: {np.mean(accuracies):.4f}")
    print(f"Best Fold Accuracy: {best_acc:.4f}")

    # Save the best model for later explainability
    import joblib
    joblib.dump(best_model, "best_xgb_model.pkl")
    print("Best model saved to best_xgb_model.pkl")

if __name__ == "__main__":
    main()
