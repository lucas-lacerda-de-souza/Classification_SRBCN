
Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms through Artificial Intelligence
----------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Dependencies:
    torch>=2.8.0
    torchvision>=0.23.0
    timm>=1.0.0
    pandas>=2.2.0
    numpy>=1.26.0
    matplotlib>=3.10.0
    seaborn>=0.13.2
    scikit-learn>=1.7.0
    pillow>=11.0.0
    tqdm>=4.67.0
    openpyxl>=3.1.5
"""

# =========================
# IMPORTS
# =========================
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


CLASS_NAMES = [
    "Haematolymphoid",
    "Mesenchymal",
    "Neuroectodermal-Neural-Crest",
    "Melanocytic"
]

NUM_CLASSES = 4


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.items = []

        required_cols = ["Classe", "CaseID"]

        for col in required_cols:
            if col not in dataframe.columns:
                raise ValueError(f"Excel file must contain '{col}' column.")

        self.clinical_cols = [
            c for c in dataframe.columns
            if c not in ["Classe", "CaseID"]
        ]

        if not self.clinical_cols:
            raise ValueError("No clinical or morphometric features found.")

        for _, row in dataframe.iterrows():
            class_label = int(row["Classe"])
            case_id = str(row["CaseID"])

            class_dir = os.path.join(self.image_dir, str(class_label))
            case_dir = os.path.join(class_dir, case_id)

            if not os.path.isdir(case_dir):
                continue

            clinical_vec = [
                float(row[c]) if pd.notna(row[c]) else 0.0
                for c in self.clinical_cols
            ]

            for root, _, files in os.walk(case_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.items.append({
                            "patch_path": os.path.join(root, f),
                            "clinical": clinical_vec,
                            "label": class_label,
                            "case_id": case_id
                        })

        print(f"Loaded {len(self.items)} patches from {image_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]

        image = Image.open(sample["patch_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor(sample["clinical"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return image, clinical, label


class MultimodalXception(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=4):
        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0
        )

        self.image_feature_dim = self.backbone.num_features

        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        image_features = self.backbone(image)
        clinical_features = self.clinical_net(clinical_data)

        combined = torch.cat(
            (image_features, clinical_features),
            dim=1
        )

        return self.classifier(combined)


def evaluate_model(model, loader, device, results_dir, split_name="test"):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for imgs, clinical, labels in tqdm(loader, desc=f"Evaluating {split_name}"):
            imgs = imgs.to(device)
            clinical = clinical.to(device)

            outputs = model(imgs, clinical)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    try:
        auc_macro = roc_auc_score(
            y_true_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )
    except Exception:
        auc_macro = np.nan

    try:
        auc_weighted = roc_auc_score(
            y_true_bin,
            y_prob,
            average="weighted",
            multi_class="ovr"
        )
    except Exception:
        auc_weighted = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision Weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall Weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC Macro OvR": auc_macro,
        "AUROC Weighted OvR": auc_weighted
    }

    pd.DataFrame([metrics]).to_csv(
        os.path.join(results_dir, f"{split_name}_metrics.csv"),
        index=False
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    with open(os.path.join(results_dir, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(report)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title(f"{split_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{split_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))

    for i, class_name in enumerate(CLASS_NAMES):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} AUC={roc_auc:.3f}")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name.capitalize()} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{split_name}_roc_curve.png"), dpi=300)
    plt.close()

    return metrics


def main():
    train_dir = "data/train"
    val_dir = "data/val"
    test_dir = "data/test"

    results_dir = "results/multimodal_xception_srbcn"
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_excel(os.path.join(train_dir, "clinical_data_train.xlsx"))
    val_df = pd.read_excel(os.path.join(val_dir, "clinical_data_val.xlsx"))
    test_df = pd.read_excel(os.path.join(test_dir, "clinical_data_test.xlsx"))

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = MultimodalDataset(train_df, train_dir, transform)
    val_dataset = MultimodalDataset(val_df, val_dir, eval_transform)
    test_dataset = MultimodalDataset(test_df, test_dir, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    clinical_input_dim = len(train_dataset.clinical_cols)

    model = MultimodalXception(
        clinical_input_dim=clinical_input_dim,
        num_classes=NUM_CLASSES
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    labels = torch.tensor(
        [sample["label"] for sample in train_dataset.items],
        dtype=torch.long
    )

    label_counts = torch.bincount(labels, minlength=NUM_CLASSES)
    class_weights = len(labels) / (NUM_CLASSES * label_counts.float())
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    history = []

    for epoch in range(100):
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0

        for imgs, clinical, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/100"):
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs, clinical)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs = imgs.to(device)
                clinical = clinical.to(device)
                labels = labels.to(device)

                outputs = model(imgs, clinical)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })

        print(
            f"[{epoch + 1}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(results_dir, "best_multimodal_xception_srbcn.pt")
            )

    pd.DataFrame(history).to_excel(
        os.path.join(results_dir, "training_history.xlsx"),
        index=False
    )

    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(
        model,
        val_loader,
        device,
        results_dir,
        split_name="val"
    )

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        results_dir,
        split_name="test"
    )

    pd.DataFrame([{
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()}
    }]).to_csv(
        os.path.join(results_dir, "final_summary_metrics.csv"),
        index=False
    )


if __name__ == "__main__":
    main()
```
