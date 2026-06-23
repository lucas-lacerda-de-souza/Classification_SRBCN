
Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms through Artificial Intelligence
----------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Dependencies:
  - python=3.12.11
  - numpy=1.26.4
  - pandas=2.2.3
  - scikit-learn=1.7.0
  - scipy=1.15.3
  - matplotlib=3.10.3
  - openpyxl=3.1.5
  - joblib=1.5.1
  - tqdm=4.67.1
  - pip
  - pip:
  - xgboost==3.0.2
   - lifelines==0.30.0
   - shap==0.48.0
"""

# =========================
# IMPORTS
# =========================
# Install if needed:
# pip install pandas numpy scikit-learn xgboost lifelines shap matplotlib openpyxl

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Input and output
# ------------------------------------------------------------

INPUT_FILE = r"./data/survival/survival_dataset_srbcn.xlsx"

OUTPUT_DIR = r"./results/xgboost_survival_srbcn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_excel(INPUT_FILE)

# ------------------------------------------------------------
# 2. Standardise column names
# ------------------------------------------------------------

df = df.rename(columns={
    "Outcome (month)": "time",
    "Outcome months": "time",
    "Overall survival": "time",
    "Survival time": "time",
    "Status": "status",
    "Vital status": "status",
    "Group": "group",
    "Tumour group": "group",
    "Diagnosis": "diagnosis",
    "Age": "age",
    "Sex": "sex",
    "Tumour location": "location",
    "Anatomical location": "location",
    "Nucleus: Area": "nucleus_area",
    "Nucleus: Perimeter": "nucleus_perimeter",
    "Nucleus: Circularity": "nucleus_circularity",
    "Nucleus: Eccentricity": "nucleus_eccentricity"
})

# ------------------------------------------------------------
# 3. Prepare survival outcome
# ------------------------------------------------------------

status_map = {
    "Dead": 1,
    "dead": 1,
    "DEAD": 1,
    "Death": 1,
    "death": 1,
    "Died": 1,
    "died": 1,
    "1": 1,
    1: 1,

    "Alive": 0,
    "alive": 0,
    "ALIVE": 0,
    "Living": 0,
    "living": 0,
    "Censored": 0,
    "censored": 0,
    "0": 0,
    0: 0
}

df["event"] = df["status"].map(status_map)
df["time"] = pd.to_numeric(df["time"], errors="coerce")

df = df.dropna(subset=["time", "event"])
df = df[df["time"] > 0].copy()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# ------------------------------------------------------------
# 4. Standardise SRBCN group names
# ------------------------------------------------------------

group_aliases = {
    "Hematolymphoid": "Haematolymphoid",
    "Haematolymphoid tumours": "Haematolymphoid",
    "Hematolymphoid tumours": "Haematolymphoid",

    "Mesenchymal tumours": "Mesenchymal",
    "Mesenchymal neoplasms": "Mesenchymal",

    "Neuroectodermal": "Neuroectodermal-Neural-Crest",
    "Neural crest": "Neuroectodermal-Neural-Crest",
    "Neural crest-derived": "Neuroectodermal-Neural-Crest",
    "Neuroectodermal and neural crest-derived": "Neuroectodermal-Neural-Crest",
    "Neuroectodermal/neural crest": "Neuroectodermal-Neural-Crest",

    "Melanocytic neoplasms": "Melanocytic",
    "Melanocytic tumours": "Melanocytic"
}

if "group" in df.columns:
    df["group"] = df["group"].replace(group_aliases)

# ------------------------------------------------------------
# 5. Define model features
# ------------------------------------------------------------

candidate_features = [
    "age",
    "sex",
    "group",
    "diagnosis",
    "location",
    "nucleus_area",
    "nucleus_perimeter",
    "nucleus_circularity",
    "nucleus_eccentricity"
]

candidate_features = [c for c in candidate_features if c in df.columns]

numeric_features = [
    "age",
    "nucleus_area",
    "nucleus_perimeter",
    "nucleus_circularity",
    "nucleus_eccentricity"
]

numeric_features = [c for c in numeric_features if c in candidate_features]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_features = [
    c for c in candidate_features
    if c not in numeric_features
]

model_df = df[["time", "event"] + candidate_features].dropna().copy()

# Merge rare categories
for col in categorical_features:
    counts = model_df[col].value_counts()
    rare_levels = counts[counts < 5].index
    model_df[col] = model_df[col].replace(rare_levels, "Other")

# One-hot encode categorical variables
X = pd.get_dummies(
    model_df[candidate_features],
    columns=categorical_features,
    drop_first=True
)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

time = model_df["time"].astype(float).values
event = model_df["event"].astype(int).values

# XGBoost survival:cox convention:
# positive time = observed event
# negative time = censored observation
y = np.where(event == 1, time, -time)

# ------------------------------------------------------------
# 6. Train / validation / test split
# ------------------------------------------------------------

X_train_val, X_test, y_train_val, y_test, time_train_val, time_test, event_train_val, event_test = train_test_split(
    X,
    y,
    time,
    event,
    test_size=0.20,
    random_state=42,
    stratify=event
)

X_train, X_val, y_train, y_val, time_train, time_val, event_train, event_val = train_test_split(
    X_train_val,
    y_train_val,
    time_train_val,
    event_train_val,
    test_size=0.20,
    random_state=42,
    stratify=event_train_val
)

# ------------------------------------------------------------
# 7. Standardisation
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ------------------------------------------------------------
# 8. Custom C-index scorer
# ------------------------------------------------------------

def xgb_survival_cindex(estimator, X_data, y_data):
    pred_risk = estimator.predict(X_data)

    observed_time = np.abs(y_data)
    observed_event = (y_data > 0).astype(int)

    return concordance_index(
        observed_time,
        -pred_risk,
        observed_event
    )

# ------------------------------------------------------------
# 9. Hyperparameter optimisation
# ------------------------------------------------------------

xgb_model = XGBRegressor(
    objective="survival:cox",
    eval_metric="cox-nloglik",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 0.1]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring=xgb_survival_cindex,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_

best_params = pd.DataFrame([grid_search.best_params_])
best_params.to_excel(
    f"{OUTPUT_DIR}/best_xgboost_survival_parameters.xlsx",
    index=False
)

# ------------------------------------------------------------
# 10. Model evaluation
# ------------------------------------------------------------

train_risk = best_model.predict(X_train_scaled)
val_risk = best_model.predict(X_val_scaled)
test_risk = best_model.predict(X_test_scaled)

train_cindex = concordance_index(time_train, -train_risk, event_train)
val_cindex = concordance_index(time_val, -val_risk, event_val)
test_cindex = concordance_index(time_test, -test_risk, event_test)

performance_table = pd.DataFrame({
    "Cohort": ["Training", "Validation", "Test"],
    "C-index": [train_cindex, val_cindex, test_cindex],
    "N": [len(X_train_scaled), len(X_val_scaled), len(X_test_scaled)],
    "Events": [
        int(event_train.sum()),
        int(event_val.sum()),
        int(event_test.sum())
    ]
})

performance_table.to_excel(
    f"{OUTPUT_DIR}/xgboost_survival_performance.xlsx",
    index=False
)

# ------------------------------------------------------------
# 11. Risk stratification
# ------------------------------------------------------------

test_results = pd.DataFrame({
    "time": time_test,
    "event": event_test,
    "risk_score": test_risk
})

risk_cutoff = np.median(test_results["risk_score"])

test_results["risk_group"] = np.where(
    test_results["risk_score"] >= risk_cutoff,
    "High-risk",
    "Low-risk"
)

test_results.to_excel(
    f"{OUTPUT_DIR}/xgboost_test_risk_scores.xlsx",
    index=False
)

# Kaplan-Meier curve by risk group
kmf = KaplanMeierFitter()

plt.figure(figsize=(7, 6))
ax = plt.gca()

for group_name, group_df in test_results.groupby("risk_group"):
    kmf.fit(
        durations=group_df["time"],
        event_observed=group_df["event"],
        label=group_name
    )
    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        linewidth=2
    )

low = test_results[test_results["risk_group"] == "Low-risk"]
high = test_results[test_results["risk_group"] == "High-risk"]

lr = logrank_test(
    low["time"],
    high["time"],
    event_observed_A=low["event"],
    event_observed_B=high["event"]
)

p_text = "Log-rank p < 0.001" if lr.p_value < 0.001 else f"Log-rank p = {lr.p_value:.3f}"

ax.text(
    0.04,
    0.08,
    p_text,
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85)
)

ax.set_xlabel("Time (months)", fontsize=12)
ax.set_ylabel("Overall survival probability", fontsize=12)
ax.set_title("XGBoost Survival Risk Stratification for SRBCNs", fontsize=14)
ax.legend(frameon=False)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_risk_groups_km.png",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_risk_groups_km.pdf",
    bbox_inches="tight"
)
plt.close()

risk_logrank_table = pd.DataFrame({
    "Comparison": ["High-risk vs Low-risk"],
    "Cut-off": [f"Median risk score = {risk_cutoff:.4f}"],
    "Log-rank p-value": [lr.p_value]
})

risk_logrank_table.to_excel(
    f"{OUTPUT_DIR}/xgboost_risk_group_logrank.xlsx",
    index=False
)

# ------------------------------------------------------------
# 12. SHAP analysis
# ------------------------------------------------------------

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test_scaled,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_shap_summary.png",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_shap_summary.pdf",
    bbox_inches="tight"
)
plt.close()

plt.figure(figsize=(9, 6))
shap.summary_plot(
    shap_values,
    X_test_scaled,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_shap_bar.png",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    f"{OUTPUT_DIR}/xgboost_survival_shap_bar.pdf",
    bbox_inches="tight"
)
plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": X_test_scaled.columns,
    "Mean absolute SHAP value": mean_abs_shap
}).sort_values(
    "Mean absolute SHAP value",
    ascending=False
)

shap_importance.to_excel(
    f"{OUTPUT_DIR}/xgboost_survival_shap_importance.xlsx",
    index=False
)

# ------------------------------------------------------------
# 13. Save all results into one Excel file
# ------------------------------------------------------------

with pd.ExcelWriter(f"{OUTPUT_DIR}/xgboost_survival_all_results.xlsx") as writer:
    performance_table.to_excel(writer, sheet_name="Performance", index=False)
    best_params.to_excel(writer, sheet_name="Best parameters", index=False)
    test_results.to_excel(writer, sheet_name="Test risk scores", index=False)
    risk_logrank_table.to_excel(writer, sheet_name="Risk log-rank", index=False)
    shap_importance.to_excel(writer, sheet_name="SHAP importance", index=False)

print("\nXGBoost survival analysis for SRBCNs completed.")
print(f"Results saved in: {OUTPUT_DIR}")

print("\nPerformance:")
print(performance_table)

print("\nBest parameters:")
print(best_params)

print("\nRisk-group log-rank:")
print(risk_logrank_table)

print("\nTop SHAP features:")
print(shap_importance.head(15))
```
