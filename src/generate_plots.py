import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
from config import OUTPUT_DIR, DATA_PROC
import joblib

def plot_dependence(explainer, shap_values, X, feature, interaction_feature=None, save_path=None):
    fig = plt.figure(figsize=(6,4))
    shap.dependence_plot(ind=feature, shap_values=shap_values, features=X, interaction_index=interaction_feature, show=False)
    if save_path:
        plt.tight_layout()
        fig.savefig(save_path)
    plt.close(fig)

def plot_force_for_instance(explainer, shap_values, X_row, feature_names, outpath):
    # use shap.force_plot and save as png via matplotlib
    fp = shap.force_plot(explainer.expected_value, shap_values, X_row, feature_names=feature_names, matplotlib=True, show=False)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Plotting helpers module")
