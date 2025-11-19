# interpretable-ml-shap-regression-project
A complete end-to-end Interpretable Machine Learning project using SHAP (SHapley Additive exPlanations) to analyze complex regression models. Includes preprocessing, XGBoost and DNN training, global feature importance, local SHAP force plots, dependence plots, and a fully automated pipeline.


# Interpretable ML: SHAP deep dive

## Purpose
Train two high-performance regression models (XGBoost, Keras DNN), compute SHAP explanations, produce global & local analyses, and deliver textual insights.

## Quickstart (Windows PowerShell)
1. Create venv: `python -m venv venv`
2. Activate: `.\venv\Scripts\Activate.ps1`
3. Install deps: `pip install -r requirements.txt`
4. Place dataset in `data/raw/dataset.csv` (set target name in `src/preprocess.py`)
5. Run pipeline:
6. Use `notebooks/final_report.ipynb` to create final deliverable and `submission/submission.txt`.

## Deliverables
- Preprocessing summary, metrics (R², RMSE)
- Global SHAP importance comparisons
- Local analyses for 3 instances (max err, min err, avg)
- At least 2 SHAP dependence plots
- All runnable code in `src/`

