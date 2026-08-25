from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import shap


def plot_shap_summary(model, features: pd.DataFrame, output_path: str | Path | None = None) -> None:
    """Create a SHAP summary plot for an already-fitted tree model.

    The caller must supply the exact feature matrix used for the model.
    This module intentionally performs no implicit model or data loading.
    """
    if features.empty:
        raise ValueError("features must not be empty")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    shap.summary_plot(shap_values, features, show=output_path is None)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
