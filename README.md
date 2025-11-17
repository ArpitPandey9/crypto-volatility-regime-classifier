# Crypto Volatility Regime Classifier

---

## 🚀 Project Overview

This project implements a **state-of-the-art machine learning pipeline** to classify volatility regimes (`Low`, `Medium`, `High`) for multiple cryptocurrencies based on advanced technical features.

Using **XGBoost**, robust **walk-forward validation**, and **SHAP explainability**, the model predicts the short-term volatility environment—crucial for risk management and strategy design in crypto quantitative finance.

---

## 📊 Key Features

- **Multi-crypto data**: BTC, ETH, LTC, BNB, SOL price and volume history (3 years, daily).
- **Rich feature engineering**: Returns, rolling volatility, momentum, RSI, MACD, Bollinger Bands width, and moving averages.
- **Target variable**: Volatility regime classes derived from BTC’s 7-day rolling volatility quantiles.
- **Walk-forward cross-validation** simulating real-world rolling retraining & testing.
- **XGBoost classifier** tuned with subsampling and regularization for performance & generalization.
- **SHAP (SHapley Additive exPlanations)** used for detailed model interpretability and feature importance analysis.
- Modular and clean **Python code** organized into reusable components for feature generation, modeling, and explainability.

---

## 📸 Visualizations

### 1. SHAP Summary Plot
![SHAP Summary Plot]

<img width="627" height="680" alt="image" src="https://github.com/user-attachments/assets/ece7032b-196b-4439-8c6c-4adce6a82901" />


*This plot shows the SHAP interaction values, highlighting the impact of features on model output.*

---

### 2. XGBoost Feature Importance

![XGBoost Feature Importance]<img width="820" height="455" alt="image" src="https://github.com/user-attachments/assets/3a6d7497-3508-4a47-8a20-d34e67de53e1" />


*Bar chart showing the relative importance of the top 15 features used by the model.*

---

## 🛠️ Tech Stack & Tools

- **Python 3.8+**
- Data: `yfinance`, `pandas`, `numpy`
- Feature engineering: `ta` (technical analysis)
- Machine learning: `xgboost`, `scikit-learn`
- Model explainability: `shap`
- Visualization: `matplotlib`, `seaborn`
- Environment management: `venv` or `conda`

---

## 📁 Repository Structure

├── data/ # (optional) raw or processed datasets
├── notebooks/ # Jupyter notebooks for EDA & prototyping
├── src/ # Source code modules
│ ├── features.py # Feature engineering functions
│ ├── model.py # Model training, evaluation, walk-forward CV
│ ├── shap_analysis.py # SHAP explainability and visualization
├── requirements.txt # Python dependencies
├── README.md # This documentation
