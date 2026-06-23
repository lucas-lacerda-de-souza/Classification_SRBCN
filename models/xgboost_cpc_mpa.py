
Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms through Artificial Intelligence
----------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Dependencies:
    python>=3.12.0
    pandas>=2.2.0
    numpy>=1.26.0
    matplotlib>=3.10.0
    scikit-learn>=1.7.0
    xgboost>=3.0.0
    shap>=0.48.0
    openpyxl>=3.1.5
"""

# =========================
# IMPORTS
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)

# ================================================================
# 1. INPUT AND OUTPUT
# ================================================================

INPUT_FILE = r"./supplementary_data/supplementary_table_8.xlsx"

OUTPUT_DIR = r"./results/xgboost_multiclass_srbcn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_FILE = os.path.join(OUTPUT_DIR, "xgboost_multiclass_metrics.xlsx")
CLASSIFICATION_REPORT_FILE = os.path.join(OUTPUT_DIR, "classification_report.xlsx")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "xgboost_multiclass_predictions.xlsx")

BARPLOT_FILE = os.path.join(OUTPUT_DIR, "xgboost_metrics_barplot.png")
CONFUSION_MATRIX_FILE = os.path.join(OUTPUT_DIR, "confusion_matrix_absolute.png")
CONFUSION_MATRIX_PERCENT_FILE = os.path.join(OUTPUT_DIR, "confusion_matrix_percent.png")
ROC_FILE = os.path.join(OUTPUT_DIR, "roc_curve_multiclass.png")

SHAP_SUMMARY_FILE_PREFIX = os.path.join(OUTPUT_DIR, "shap_summary_class")
SHAP_BAR_FILE_PREFIX = os.path.join(OUTPUT_DIR, "shap_bar_class")

# ================================================================
# 2. LOAD DATASET
# ================================================================

data = pd.read_excel(INPUT_FILE)

TARGET_COLUMN = "Group"

FEATURES = [
    "Age",
    "Sex",
    "Anatomical location",
    "Nucleus: Area",
    "Nucleus: Perimeter",
    "Nucleus: Circularity",
    "Nucleus: Eccentricity"
]

data = data[[TARGET_COLUMN] + FEATURES].copy()
data = data.dropna()

# ================================================================
# 3. EXPECTED SRBCN GROUPS
# ================================================================

EXPECTED_GROUPS = [
    "Haematolymphoid",
    "Mesenchymal",
    "Neuroectodermal-Neural-Crest",
    "Melanocytic"
]

# Optional: standardise names if needed
GROUP_ALIASES = {
    "Hematolymphoid": "Haematolymphoid",
    "Neuroectodermal": "Neuroectodermal-Neural-Crest",
    "Neuroectodermal and neural crest-derived": "Neuroectodermal-Neural-Crest",
    "Neuroectodermal-Neural Crest": "Neuroectodermal-Neural-Crest",
    "Melanocytic neoplasms": "Melanocytic",
    "Mesenchymal tumours": "Mesenchymal",
    "Haematolymphoid tumours": "Haematolymphoid"
}

data[TARGET_COLUMN] = data[TARGET_COLUMN].replace(GROUP_ALIASES)

# ================================================================
# 4. DEFINE NUMERIC AND CATEGORICAL VARIABLES
# ================================================================

numeric_features = [
    "Age",
    "Nucleus: Area",
    "Nucleus: Perimeter",
    "Nucleus: Circularity",
    "Nucleus: Eccentricity"
]

categorical_features = [
    "Sex",
    "Anatomical location"
]

data[numeric_features] = data[numeric_features].apply(
    pd.to_numeric,
    errors="coerce"
)

data = data.dropna()

# ================================================================
# 5. ENCODE TARGET CLASSES
# ================================================================

label_encoder = LabelEncoder()
data["Group_encoded"] = label_encoder.fit_transform(data[TARGET_COLUMN])

class_names = label_encoder.classes_
num_classes = len(class_names)

print("Detected classes:")
for i, c in enumerate(class_names):
    print(f"{i}: {c}")

if num_classes != 4:
    print(f"Warning: expected 4 classes, but found {num_classes}: {class_names}")

X = data[FEATURES]
y = data["Group_encoded"]

# ================================================================
# 6. TRAIN/TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=123,
    stratify=y
)

# ================================================================
# 7. PREPROCESSING PIPELINE
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ================================================================
# 8. XGBOOST MULTICLASS MODEL
# ================================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="mlogloss",
    random_state=123
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

# ================================================================
# 9. PREDICTIONS
# ================================================================

y_pred = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)

# ================================================================
# 10. GLOBAL METRICS
# ================================================================

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

auc_macro_ovr = roc_auc_score(
    y_test_bin,
    y_pred_prob,
    average="macro",
    multi_class="ovr"
)

auc_weighted_ovr = roc_auc_score(
    y_test_bin,
    y_pred_prob,
    average="weighted",
    multi_class="ovr"
)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision Macro",
        "Recall Macro",
        "F1 Macro",
        "Precision Weighted",
        "Recall Weighted",
        "F1 Weighted",
        "AUC Macro OvR",
        "AUC Weighted OvR",
        "Cohen Kappa",
        "Matthews Correlation Coefficient"
    ],
    "Value": [
        accuracy,
        balanced_accuracy,
        precision_macro,
        recall_macro,
        f1_macro,
        precision_weighted,
        recall_weighted,
        f1_weighted,
        auc_macro_ovr,
        auc_weighted_ovr,
        kappa,
        mcc
    ]
})

print("\nGlobal Metrics:")
print(metrics_df.round(4))

metrics_df.to_excel(METRICS_FILE, index=False)

# ================================================================
# 11. CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
report_df.to_excel(CLASSIFICATION_REPORT_FILE)

# ================================================================
# 12. SAVE PREDICTIONS
# ================================================================

predictions_df = X_test.copy()
predictions_df["True_Group"] = label_encoder.inverse_transform(y_test)
predictions_df["Predicted_Group"] = label_encoder.inverse_transform(y_pred)

for i, class_name in enumerate(class_names):
    predictions_df[f"Probability_{class_name}"] = y_pred_prob[:, i]

predictions_df.to_excel(PREDICTIONS_FILE, index=False)

# ================================================================
# 13. METRIC BAR PLOT
# ================================================================

plt.figure(figsize=(13, 6))
plt.bar(metrics_df["Metric"], metrics_df["Value"], edgecolor="black")
plt.ylim(-1, 1)
plt.title("XGBoost Multiclass Model Performance for SRBCNs", fontsize=16, fontweight="bold")
plt.ylabel("Value")
plt.xticks(rotation=35, ha="right")

for i, value in enumerate(metrics_df["Value"]):
    plt.text(i, value + 0.03, f"{value:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(BARPLOT_FILE, dpi=300)
plt.close()

# ================================================================
# 14. CONFUSION MATRIX - ABSOLUTE VALUES
# ================================================================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Greys", values_format="d")
plt.title("Confusion Matrix - Absolute Values")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
plt.close()

# ================================================================
# 15. CONFUSION MATRIX - PERCENTAGE
# ================================================================

cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
cm_percent = np.nan_to_num(cm_percent)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_percent, cmap="Greys")

ax.set_title("Confusion Matrix - Percentage")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))

ax.set_xticklabels(class_names, rotation=35, ha="right")
ax.set_yticklabels(class_names)

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j,
            i,
            f"{cm_percent[i, j]:.1f}%",
            ha="center",
            va="center"
        )

plt.colorbar(im)
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PERCENT_FILE, dpi=300)
plt.close()

# ================================================================
# 16. ROC CURVE - MULTICLASS ONE-VS-REST
# ================================================================

plt.figure(figsize=(8, 7))

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} AUC = {roc_auc:.3f}"
    )

plt.plot([0, 1], [0, 1], linestyle=":", linewidth=1)

plt.title("ROC Curves - Multiclass One-vs-Rest", fontsize=16, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_FILE, dpi=300)
plt.close()

# ================================================================
# 17. SHAP ANALYSIS
# ================================================================

X_train_processed = pipeline.named_steps["preprocessor"].transform(X_train)

feature_names_numeric = numeric_features

feature_names_categorical = pipeline.named_steps["preprocessor"] \
    .named_transformers_["categorical"] \
    .get_feature_names_out(categorical_features)

feature_names_processed = list(feature_names_numeric) + list(feature_names_categorical)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names_processed
)

xgb_model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train_processed_df)

# ================================================================
# 18. SHAP SUMMARY AND BAR PLOTS FOR EACH CLASS
# ================================================================

for class_index, class_name in enumerate(class_names):

    safe_class_name = str(class_name).replace("/", "_").replace(" ", "_")

    if isinstance(shap_values, list):
        shap_values_class = shap_values[class_index]
    else:
        shap_values_class = shap_values[:, :, class_index]

    shap.summary_plot(
        shap_values_class,
        X_train_processed_df,
        show=False
    )

    plt.title(f"SHAP Summary Plot - {class_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SHAP_SUMMARY_FILE_PREFIX}_{safe_class_name}.png", dpi=300)
    plt.close()

    shap.summary_plot(
        shap_values_class,
        X_train_processed_df,
        plot_type="bar",
        show=False
    )

    plt.title(f"SHAP Feature Importance - {class_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{SHAP_BAR_FILE_PREFIX}_{safe_class_name}.png", dpi=300)
    plt.close()

# ================================================================
# 19. FINAL MESSAGE
# ================================================================

print("\nXGBoost multiclass SRBCN analysis completed successfully.")
print(f"Results saved in: {OUTPUT_DIR}")
```
