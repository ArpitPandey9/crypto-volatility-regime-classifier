from ctypes import c_short
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load model
best_model = joblib.load("best_xgb_model.pkl")

# Recreate or load the exact feature dataframe X_short you used during training
# This is a placeholder — replace with your actual feature loading method
# For example, re-run feature engineering and drop volatility_regime column

# X_short = ...

# Flatten multiindex columns if present
if isinstance(c_short.columns, pd.MultiIndex):
    c_short.columns = ['_'.join(col).strip() for col in c_short.columns.values]

c_short.columns = (
    pd.Index(c_short.columns)
      .str.replace('btc_', 'B_')
      .str.replace('eth_', 'E_')
      .str.replace('ltc_', 'L_')
      .str.replace('bnb_', 'N_')
      .str.replace('sol_', 'S_')
)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(c_short)

plt.rcParams['ytick.labelsize'] = 8  # smaller font size to avoid overlap

shap.summary_plot(shap_values, c_short, class_names=None)

plt.show()
