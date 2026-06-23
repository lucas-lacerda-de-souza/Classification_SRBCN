
Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms through Artificial Intelligence
----------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Dependencies:
    python>=3.12.0
    torch>=2.8.0
    torchvision>=0.23.0
    timm>=1.0.0
    huggingface-hub>=0.34.0
    openslide-python>=1.4.0
    pillow>=11.0.0
    numpy>=1.26.0
    pandas>=2.2.0
    matplotlib>=3.10.0
    scikit-learn>=1.7.0
    tqdm>=4.67.0
    openpyxl>=3.1.0
"""

# =========================
# IMPORTS
# =========================
import argparse
import json
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openslide
import timm
import torch
import torch.nn as nn

from huggingface_hub import login
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =========================
# FIXED 4-CLASS SRBCN SETUP
# =========================

TARGET_CLASSES = [
    "Haematolymphoid",
    "Mesenchymal",
    "Neuroectodermal-Neural-Crest",
    "Melanocytic",
]

CLASS_MAP = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

TARGET_CLASS_ALIASES = {
    "Haematolymphoid": [
        "Haematolymphoid",
        "Hematolymphoid",
        "Lymphoid",
        "Burkitt lymphoma",
        "BL",
        "Diffuse large B-cell lymphoma",
        "DLBCL",
        "Lymphoblastic lymphoma",
        "Plasmablastic lymphoma",
        "PBL",
        "Follicular lymphoma",
        "FL",
        "Reactive follicular hyperplasia",
        "RFH",
    ],
    "Mesenchymal": [
        "Mesenchymal",
        "Sarcoma",
        "Rhabdomyosarcoma",
        "RMS",
        "Ewing sarcoma",
        "Ewing",
        "Ewing's sarcoma",
        "Small round cell sarcoma",
    ],
    "Neuroectodermal-Neural-Crest": [
        "Neuroectodermal",
        "Neural crest",
        "Neural crest-derived",
        "Neuroectodermal and neural crest-derived",
        "Neuroblastoma",
        "Retinoblastoma",
        "PNET",
    ],
    "Melanocytic": [
        "Melanocytic",
        "Melanoma",
        "Malignant melanoma",
        "Mucosal melanoma",
        "Oral melanoma",
    ],
}


# =========================
# ARGUMENTS
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Virchow feature extraction, patch-level classification, MIL classification, and patient-level evaluation for SRBCNs."
    )

    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output_virchow")
    parser.add_argument("--virchow_model", type=str, default="paige-ai/Virchow")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patches_per_slide", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face token. Virchow requires approved access.",
    )

    return parser.parse_args()


# =========================
# CONFIG
# =========================

class Config:
    def __init__(self, args):
        self.ROOT_DIR = args.root_dir
        self.OUTPUT_DIR = args.output_dir
        self.CACHE_DIR = os.path.join(args.output_dir, "cache")
        self.PLOTS_DIR = os.path.join(args.output_dir, "plots")
        self.METRICS_DIR = os.path.join(args.output_dir, "metrics")
        self.MODELS_DIR = os.path.join(args.output_dir, "models")

        self.VIRCHOW_MODEL = args.virchow_model
        self.HF_TOKEN = args.hf_token

        self.SEED = args.seed
        self.PATCH_SIZE = args.patch_size
        self.PATCHES_PER_SLIDE = args.patches_per_slide
        self.BATCH_SIZE = args.batch_size
        self.EPOCHS = args.epochs
        self.NUM_WORKERS = args.num_workers
        self.LR = args.lr

        self.TRAIN_RATIO = args.train_ratio
        self.VAL_RATIO = args.val_ratio

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        for d in [
            self.OUTPUT_DIR,
            self.CACHE_DIR,
            self.PLOTS_DIR,
            self.METRICS_DIR,
            self.MODELS_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


# =========================
# UTILITIES
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert_to_native(obj), f, indent=4, ensure_ascii=False)


def normalize_name(txt):
    return str(txt).strip().lower().replace("_", " ").replace("-", " ")


def map_folder_to_target_class(folder_name):
    folder_norm = normalize_name(folder_name)

    for target_class, aliases in TARGET_CLASS_ALIASES.items():
        all_names = [target_class] + aliases
        all_names = [normalize_name(x) for x in all_names]

        if folder_norm in all_names:
            return target_class

    return None


# =========================
# TRANSFORM
# =========================

def build_transform(patch_size):
    return transforms.Compose(
        [
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


# =========================
# VIRCHOW
# =========================

def load_virchow(cfg):
    print("\nLoading Virchow foundation model...")

    if cfg.HF_TOKEN is not None and str(cfg.HF_TOKEN).strip():
        login(token=cfg.HF_TOKEN, add_to_git_credential=False)
    else:
        print("No Hugging Face token provided.")
        print("If the model is gated, run: export HF_TOKEN='your_token'")

    start = time.time()

    model = timm.create_model(
        f"hf_hub:{cfg.VIRCHOW_MODEL}",
        pretrained=True,
        num_classes=0,
    )

    model = model.to(cfg.DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(f"Virchow loaded in {time.time() - start:.2f}s")
    return model


class VirchowFeatureExtractor(nn.Module):
    def __init__(self, virchow_model):
        super().__init__()
        self.virchow = virchow_model

    def forward(self, x):
        output = self.virchow(x)

        if isinstance(output, dict):
            output = output.get("x", output.get("last_hidden_state"))

        if isinstance(output, (list, tuple)):
            output = output[0]

        if output.ndim == 3:
            cls_token = output[:, 0]
            patch_tokens = output[:, 1:]
            embedding = torch.cat(
                [cls_token, patch_tokens.mean(dim=1)],
                dim=-1,
            )
        else:
            embedding = output

        return embedding


# =========================
# DATA DISCOVERY
# =========================

def find_svs_recursively(class_dir):
    class_dir = Path(class_dir)
    svs_files = [str(p) for p in class_dir.rglob("*.svs") if p.is_file()]
    svs_files.sort()
    return svs_files


def build_case_and_patient_id(class_dir, slide_path):
    class_dir = Path(class_dir)
    slide_path = Path(slide_path)

    rel_parts = slide_path.relative_to(class_dir).parts

    if len(rel_parts) >= 2:
        case_id = rel_parts[0]
    else:
        case_id = slide_path.stem

    patient_id = case_id
    return case_id, patient_id


def load_data(cfg):
    print("\nLoading SRBCN dataset recursively...\n")

    samples = []
    skipped_folders = []

    discovered_dirs = [
        c for c in sorted(os.listdir(cfg.ROOT_DIR))
        if os.path.isdir(os.path.join(cfg.ROOT_DIR, c))
    ]

    for folder_name in tqdm(discovered_dirs, desc="Folders", dynamic_ncols=True, bar_format=cfg.TQDM_BAR):
        class_path = os.path.join(cfg.ROOT_DIR, folder_name)
        target_class = map_folder_to_target_class(folder_name)

        if target_class is None:
            skipped_folders.append(folder_name)
            continue

        svs_files = find_svs_recursively(class_path)

        for full in svs_files:
            case_id, patient_id = build_case_and_patient_id(class_path, full)
            rel = os.path.relpath(full, class_path)
            sample_id = f"{target_class}__{patient_id}__{Path(full).stem}"

            samples.append(
                {
                    "path": full,
                    "label": CLASS_MAP[target_class],
                    "label_name": target_class,
                    "original_folder": folder_name,
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "relative_path": rel,
                    "id": sample_id,
                }
            )

    distribution = dict(Counter([s["label_name"] for s in samples]))

    print(f"Total slides found: {len(samples)}")
    print("Distribution:", distribution)

    save_json(
        {
            "class_map": CLASS_MAP,
            "target_classes": TARGET_CLASSES,
            "total_slides": len(samples),
            "distribution": distribution,
            "skipped_folders": skipped_folders,
        },
        os.path.join(cfg.METRICS_DIR, "dataset_discovery.json"),
    )

    return samples, CLASS_MAP


def validate_slides(samples, cfg):
    print("\nValidating slides...\n")

    valid_samples = []
    invalid_samples = []

    for s in tqdm(samples, desc="Checking slides", dynamic_ncols=True, bar_format=cfg.TQDM_BAR):
        try:
            slide = openslide.OpenSlide(s["path"])
            _ = slide.dimensions
            slide.close()
            valid_samples.append(s)
        except Exception as e:
            invalid_samples.append(
                {
                    "path": s["path"],
                    "label": s["label"],
                    "label_name": s["label_name"],
                    "id": s["id"],
                    "error": str(e),
                }
            )

    save_json(invalid_samples, os.path.join(cfg.METRICS_DIR, "invalid_slides.json"))

    print(f"Valid slides: {len(valid_samples)}")
    print(f"Invalid slides: {len(invalid_samples)}")

    return valid_samples, invalid_samples


# =========================
# SPLIT
# =========================

def split_samples(samples, cfg):
    labels = [s["label"] for s in samples]
    class_counts = Counter(labels)

    rare_classes = [k for k, v in class_counts.items() if v < 3]

    if rare_classes:
        print("\nRare classes detected. Using random split.")
        shuffled = samples[:]
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(cfg.TRAIN_RATIO * n)
        n_val = int(cfg.VAL_RATIO * n)

        return (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )

    train_s, temp_s = train_test_split(
        samples,
        test_size=(1 - cfg.TRAIN_RATIO),
        stratify=labels,
        random_state=cfg.SEED,
    )

    temp_labels = [s["label"] for s in temp_s]
    val_fraction = cfg.VAL_RATIO / (1 - cfg.TRAIN_RATIO)

    val_s, test_s = train_test_split(
        temp_s,
        test_size=(1 - val_fraction),
        stratify=temp_labels,
        random_state=cfg.SEED,
    )

    return train_s, val_s, test_s


def compute_class_weights_from_samples(train_samples, num_classes, cfg):
    counts = Counter([s["label"] for s in train_samples])
    total = len(train_samples)

    weights = []

    for cls_idx in range(num_classes):
        n_cls = counts.get(cls_idx, 0)
        if n_cls == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * n_cls))

    weights = torch.tensor(weights, dtype=torch.float32)

    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes

    save_json(
        {
            "train_counts": {
                TARGET_CLASSES[i]: counts.get(i, 0)
                for i in range(num_classes)
            },
            "class_weights": {
                TARGET_CLASSES[i]: float(weights[i].item())
                for i in range(num_classes)
            },
        },
        os.path.join(cfg.METRICS_DIR, "class_weights.json"),
    )

    return weights


# =========================
# DATASET
# =========================

class DatasetSVS(Dataset):
    def __init__(self, samples, cfg, transform):
        self.items = []
        self.cfg = cfg
        self.transform = transform

        for s in samples:
            for _ in range(cfg.PATCHES_PER_SLIDE):
                self.items.append(s)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]

        try:
            slide = openslide.OpenSlide(s["path"])
            w, h = slide.dimensions

            x = random.randint(0, max(0, w - self.cfg.PATCH_SIZE))
            y = random.randint(0, max(0, h - self.cfg.PATCH_SIZE))

            patch = slide.read_region(
                (x, y),
                0,
                (self.cfg.PATCH_SIZE, self.cfg.PATCH_SIZE),
            ).convert("RGB")

            slide.close()

        except Exception:
            patch = Image.new(
                "RGB",
                (self.cfg.PATCH_SIZE, self.cfg.PATCH_SIZE),
                (255, 255, 255),
            )

        return self.transform(patch), s["label"], s["id"]


# =========================
# FEATURE EXTRACTION
# =========================

def extract_or_load(model, loader, name, cfg):
    f_path = os.path.join(cfg.CACHE_DIR, f"{name}_X.pt")
    l_path = os.path.join(cfg.CACHE_DIR, f"{name}_y.pt")
    id_path = os.path.join(cfg.CACHE_DIR, f"{name}_id.pt")

    if os.path.exists(f_path) and os.path.exists(l_path) and os.path.exists(id_path):
        print(f"\nLoading cached features [{name}]")
        X = torch.load(f_path)
        y = torch.load(l_path)
        ids = torch.load(id_path)
        return X, y, ids

    print(f"\nExtracting Virchow features [{name}]...\n")

    feats, labels, ids = [], [], []

    model.eval()

    with torch.no_grad():
        for x, y, i in tqdm(loader, desc=f"Extracting {name}", dynamic_ncols=True, bar_format=cfg.TQDM_BAR):
            f = model(x.to(cfg.DEVICE))

            feats.append(f.cpu())
            labels.append(y)
            ids.extend(i)

    X = torch.cat(feats)
    y = torch.cat(labels)

    torch.save(X, f_path)
    torch.save(y, l_path)
    torch.save(ids, id_path)

    return X, y, ids


def create_bags(X, y, ids):
    bags = defaultdict(lambda: {"X": [], "y": None})

    for i, sid in enumerate(ids):
        bags[sid]["X"].append(X[i])
        if bags[sid]["y"] is None:
            bags[sid]["y"] = int(y[i].item())

    final_bags = []

    for sid, content in bags.items():
        final_bags.append(
            (
                sid,
                torch.stack(content["X"], dim=0),
                content["y"],
            )
        )

    return final_bags


# =========================
# MODELS
# =========================

class PatchClassifier(nn.Module):
    def __init__(self, d, c):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, c),
        )

    def forward(self, x):
        return self.net(x)


class MILAttention(nn.Module):
    def __init__(self, d, c):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, c),
        )

    def forward(self, x, return_attention=False):
        A = self.attention(x)
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * x, dim=0)
        out = self.classifier(M)

        if return_attention:
            return out, A

        return out


# =========================
# METRICS
# =========================

def get_ovr_confusion_terms(y_true, y_pred, class_idx):
    y_true_bin = (np.array(y_true) == class_idx).astype(int)
    y_pred_bin = (np.array(y_pred) == class_idx).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin,
        y_pred_bin,
        labels=[0, 1],
    ).ravel()

    return tp, fn, fp, tn


def expected_calibration_error(y_true, probs, n_bins=10):
    y_true = np.array(y_true)
    probs = np.array(probs)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.abs(bin_acc - bin_conf) * np.mean(mask)

    return float(ece)


def multiclass_brier_score(y_true, probs, n_classes):
    y_onehot = np.eye(n_classes)[np.array(y_true)]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def compute_metrics(y_true, y_pred, probs, class_names):
    n_classes = len(class_names)
    labels = list(range(n_classes))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": expected_calibration_error(y_true, probs),
        "brier_score": multiclass_brier_score(y_true, probs, n_classes),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    y_bin = label_binarize(y_true, classes=labels)

    try:
        metrics["roc_auc_macro_ovr"] = float(
            roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc_macro_ovr"] = None

    try:
        metrics["roc_auc_weighted_ovr"] = float(
            roc_auc_score(y_bin, probs, multi_class="ovr", average="weighted")
        )
    except Exception:
        metrics["roc_auc_weighted_ovr"] = None

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    per_class = {}

    for i, cname in enumerate(class_names):
        tp, fn, fp, tn = get_ovr_confusion_terms(y_true, y_pred, i)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[cname] = {
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "sensitivity": float(sens),
            "specificity": float(spec),
        }

    metrics["per_class"] = per_class

    return metrics, cm


# =========================
# PLOTS
# =========================

def plot_confusion_matrix(cm, class_names, save_path, normalize=False, title="Confusion Matrix"):
    cm_plot = cm.astype(np.float64).copy()

    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sums, out=np.zeros_like(cm_plot), where=row_sums != 0)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, probs, class_names, save_path):
    n_classes = len(class_names)
    labels = list(range(n_classes))

    y_bin = label_binarize(y_true, classes=labels)

    plt.figure(figsize=(7, 6))

    plotted = False

    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cname} AUC={roc_auc:.4f}")
            plotted = True
        except Exception:
            pass

    if not plotted:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, prefix, cfg):
    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f"{prefix}_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f"{prefix}_accuracy.png"), dpi=300)
    plt.close()


# =========================
# TRAINING
# =========================

def train_patch_model(model, X_train, y_train, X_val, y_val, class_weights, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_path = os.path.join(cfg.MODELS_DIR, "best_virchow_patch_model.pt")

    for e in range(cfg.EPOCHS):
        model.train()

        perm = torch.randperm(len(X_train))
        losses = []
        preds_all = []
        true_all = []

        for i in tqdm(range(0, len(X_train), cfg.BATCH_SIZE), desc=f"Patch Epoch {e + 1}/{cfg.EPOCHS}"):
            idx = perm[i:i + cfg.BATCH_SIZE]
            xb = X_train[idx].to(cfg.DEVICE)
            yb = y_train[idx].to(cfg.DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds_all.extend(torch.argmax(out, dim=1).detach().cpu().numpy())
            true_all.extend(yb.detach().cpu().numpy())

        train_loss = float(np.mean(losses))
        train_acc = float(accuracy_score(true_all, preds_all))

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val.to(cfg.DEVICE))
            val_loss = float(loss_fn(val_logits, y_val.to(cfg.DEVICE)).item())
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_acc = float(accuracy_score(y_val.numpy(), val_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"[PATCH] Epoch {e + 1}: train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=cfg.DEVICE))
    plot_training_curves(history, "virchow_patch_training", cfg)
    save_json(history, os.path.join(cfg.METRICS_DIR, "virchow_patch_training_history.json"))

    return model, history


def evaluate_patch_model(model, X, y, ids, class_names, split, cfg):
    model.eval()

    probs_list = []
    preds_list = []
    true_list = []
    id_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X), cfg.BATCH_SIZE), desc=f"Evaluate Patch [{split}]"):
            xb = X[i:i + cfg.BATCH_SIZE].to(cfg.DEVICE)
            yb = y[i:i + cfg.BATCH_SIZE]

            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            probs_list.append(probs)
            preds_list.extend(preds.tolist())
            true_list.extend(yb.numpy().tolist())
            id_list.extend(ids[i:i + cfg.BATCH_SIZE])

    probs = np.concatenate(probs_list, axis=0)
    preds = np.array(preds_list)
    y_true = np.array(true_list)

    metrics, cm = compute_metrics(y_true, preds, probs, class_names)

    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_patch_metrics.json"))

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patch_cm.png"),
        False,
        f"{split.upper()} Patch CM",
    )

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patch_cm_norm.png"),
        True,
        f"{split.upper()} Patch CM Normalized",
    )

    plot_roc_curve(
        y_true,
        probs,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patch_roc.png"),
    )

    return metrics, y_true, preds, probs, id_list


# =========================
# PATIENT LEVEL
# =========================

def extract_patient_id(sample_id):
    parts = str(sample_id).split("__")

    if len(parts) >= 3:
        return parts[1]

    return str(sample_id)


def aggregate_to_patient_level(ids, y_true, probs):
    patient_dict = defaultdict(lambda: {"probs": [], "labels": []})

    for sid, yt, pr in zip(ids, y_true, probs):
        pid = extract_patient_id(sid)
        patient_dict[pid]["probs"].append(pr)
        patient_dict[pid]["labels"].append(int(yt))

    patient_ids = []
    patient_y_true = []
    patient_probs = []

    for pid, content in patient_dict.items():
        avg_probs = np.mean(np.stack(content["probs"], axis=0), axis=0)
        labels = content["labels"]
        final_label = Counter(labels).most_common(1)[0][0]

        patient_ids.append(pid)
        patient_y_true.append(final_label)
        patient_probs.append(avg_probs)

    patient_probs = np.array(patient_probs)
    patient_y_true = np.array(patient_y_true)
    patient_y_pred = np.argmax(patient_probs, axis=1)

    return patient_ids, patient_y_true, patient_y_pred, patient_probs


def evaluate_patient_level(ids, y_true, probs, class_names, split, cfg):
    patient_ids, patient_y_true, patient_y_pred, patient_probs = aggregate_to_patient_level(
        ids,
        y_true,
        probs,
    )

    metrics, cm = compute_metrics(
        patient_y_true,
        patient_y_pred,
        patient_probs,
        class_names,
    )

    metrics["n_patients"] = len(patient_ids)

    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_patient_level_metrics.json"))

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patient_cm.png"),
        False,
        f"{split.upper()} Patient CM",
    )

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patient_cm_norm.png"),
        True,
        f"{split.upper()} Patient CM Normalized",
    )

    plot_roc_curve(
        patient_y_true,
        patient_probs,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_patient_roc.png"),
    )

    return metrics


# =========================
# MIL
# =========================

def train_mil_model(model, train_bags, val_bags, class_weights, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_path = os.path.join(cfg.MODELS_DIR, "best_virchow_mil_model.pt")

    for e in range(cfg.EPOCHS):
        model.train()
        random.shuffle(train_bags)

        losses = []
        preds_all = []
        true_all = []

        for sid, bag_x, bag_y in tqdm(train_bags, desc=f"MIL Epoch {e + 1}/{cfg.EPOCHS}"):
            bag_x = bag_x.to(cfg.DEVICE)
            yb = torch.tensor([bag_y], dtype=torch.long, device=cfg.DEVICE)

            optimizer.zero_grad()
            out = model(bag_x).unsqueeze(0)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds_all.append(int(torch.argmax(out, dim=1).item()))
            true_all.append(int(bag_y))

        train_loss = float(np.mean(losses))
        train_acc = float(accuracy_score(true_all, preds_all))

        model.eval()
        val_losses = []
        val_preds = []
        val_true = []

        with torch.no_grad():
            for sid, bag_x, bag_y in val_bags:
                bag_x = bag_x.to(cfg.DEVICE)
                yb = torch.tensor([bag_y], dtype=torch.long, device=cfg.DEVICE)

                out = model(bag_x).unsqueeze(0)
                loss = loss_fn(out, yb)

                val_losses.append(loss.item())
                val_preds.append(int(torch.argmax(out, dim=1).item()))
                val_true.append(int(bag_y))

        val_loss = float(np.mean(val_losses))
        val_acc = float(accuracy_score(val_true, val_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"[MIL] Epoch {e + 1}: train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=cfg.DEVICE))
    plot_training_curves(history, "virchow_mil_training", cfg)
    save_json(history, os.path.join(cfg.METRICS_DIR, "virchow_mil_training_history.json"))

    return model, history


def evaluate_mil_model(model, bags, class_names, split, cfg):
    model.eval()

    probs_list = []
    preds_list = []
    true_list = []

    with torch.no_grad():
        for sid, bag_x, bag_y in tqdm(bags, desc=f"Evaluate MIL [{split}]"):
            bag_x = bag_x.to(cfg.DEVICE)

            out = model(bag_x)
            probs = torch.softmax(out, dim=0).cpu().numpy()
            pred = int(np.argmax(probs))

            probs_list.append(probs[None, :])
            preds_list.append(pred)
            true_list.append(int(bag_y))

    probs = np.concatenate(probs_list, axis=0)
    preds = np.array(preds_list)
    y_true = np.array(true_list)

    metrics, cm = compute_metrics(y_true, preds, probs, class_names)

    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_mil_metrics.json"))

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_mil_cm.png"),
        False,
        f"{split.upper()} MIL CM",
    )

    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_mil_cm_norm.png"),
        True,
        f"{split.upper()} MIL CM Normalized",
    )

    plot_roc_curve(
        y_true,
        probs,
        class_names,
        os.path.join(cfg.PLOTS_DIR, f"{split}_mil_roc.png"),
    )

    return metrics


# =========================
# FINAL SUMMARY
# =========================

def save_final_summary(
    class_names,
    class_map,
    train_s,
    val_s,
    test_s,
    invalid_samples,
    val_patch_metrics,
    test_patch_metrics,
    val_patient_metrics,
    test_patient_metrics,
    val_mil_metrics,
    test_mil_metrics,
    cfg,
):
    summary = {
        "model": "Virchow",
        "classes": class_names,
        "class_map": class_map,
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "invalid_slides": len(invalid_samples),
        "train_distribution": dict(Counter([s["label_name"] for s in train_s])),
        "val_distribution": dict(Counter([s["label_name"] for s in val_s])),
        "test_distribution": dict(Counter([s["label_name"] for s in test_s])),
        "val_patch_metrics": val_patch_metrics,
        "test_patch_metrics": test_patch_metrics,
        "val_patient_metrics": val_patient_metrics,
        "test_patient_metrics": test_patient_metrics,
        "val_mil_metrics": val_mil_metrics,
        "test_mil_metrics": test_mil_metrics,
    }

    save_json(summary, os.path.join(cfg.OUTPUT_DIR, "final_summary_virchow.json"))


# =========================
# MAIN
# =========================

def main():
    args = parse_args()
    cfg = Config(args)

    total_start = time.time()
    set_seed(cfg.SEED)

    print("\nVirchow pipeline for four-class SRBCN classification\n")
    print(f"Device: {cfg.DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = build_transform(cfg.PATCH_SIZE)

    samples, class_map = load_data(cfg)

    if len(samples) == 0:
        raise RuntimeError("No .svs slides were found for the SRBCN target classes.")

    samples, invalid_samples = validate_slides(samples, cfg)

    if len(samples) == 0:
        raise RuntimeError("No valid slides remained after validation.")

    train_s, val_s, test_s = split_samples(samples, cfg)

    print(f"\nTrain: {len(train_s)} | Validation: {len(val_s)} | Test: {len(test_s)}")

    class_weights = compute_class_weights_from_samples(
        train_s,
        num_classes=len(TARGET_CLASSES),
        cfg=cfg,
    )

    virchow_base = load_virchow(cfg)
    virchow = VirchowFeatureExtractor(virchow_base).to(cfg.DEVICE)
    virchow.eval()

    train_loader = DataLoader(
        DatasetSVS(train_s, cfg, transform),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
    )

    val_loader = DataLoader(
        DatasetSVS(val_s, cfg, transform),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
    )

    test_loader = DataLoader(
        DatasetSVS(test_s, cfg, transform),
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
    )

    Xtr, ytr, idtr = extract_or_load(virchow, train_loader, "train_virchow", cfg)
    Xv, yv, idv = extract_or_load(virchow, val_loader, "val_virchow", cfg)
    Xte, yte, idte = extract_or_load(virchow, test_loader, "test_virchow", cfg)

    class_names = TARGET_CLASSES

    print("\nTraining Virchow patch classifier...\n")

    patch_model = PatchClassifier(
        d=Xtr.shape[1],
        c=len(class_names),
    ).to(cfg.DEVICE)

    patch_model, _ = train_patch_model(
        patch_model,
        Xtr,
        ytr,
        Xv,
        yv,
        class_weights,
        cfg,
    )

    print("\nEvaluating Virchow patch classifier...\n")

    val_patch_metrics, val_y_true, val_y_pred, val_probs, val_ids = evaluate_patch_model(
        patch_model,
        Xv,
        yv,
        idv,
        class_names,
        "val_virchow",
        cfg,
    )

    test_patch_metrics, test_y_true, test_y_pred, test_probs, test_ids = evaluate_patch_model(
        patch_model,
        Xte,
        yte,
        idte,
        class_names,
        "test_virchow",
        cfg,
    )

    print("\nEvaluating patient-level predictions...\n")

    val_patient_metrics = evaluate_patient_level(
        val_ids,
        val_y_true,
        val_probs,
        class_names,
        "val_virchow",
        cfg,
    )

    test_patient_metrics = evaluate_patient_level(
        test_ids,
        test_y_true,
        test_probs,
        class_names,
        "test_virchow",
        cfg,
    )

    print("\nCreating MIL bags...\n")

    train_bags = create_bags(Xtr, ytr, idtr)
    val_bags = create_bags(Xv, yv, idv)
    test_bags = create_bags(Xte, yte, idte)

    print("\nTraining Virchow MIL attention model...\n")

    mil_model = MILAttention(
        d=Xtr.shape[1],
        c=len(class_names),
    ).to(cfg.DEVICE)

    mil_model, _ = train_mil_model(
        mil_model,
        train_bags,
        val_bags,
        class_weights,
        cfg,
    )

    print("\nEvaluating Virchow MIL model...\n")

    val_mil_metrics = evaluate_mil_model(
        mil_model,
        val_bags,
        class_names,
        "val_virchow",
        cfg,
    )

    test_mil_metrics = evaluate_mil_model(
        mil_model,
        test_bags,
        class_names,
        "test_virchow",
        cfg,
    )

    save_final_summary(
        class_names,
        class_map,
        train_s,
        val_s,
        test_s,
        invalid_samples,
        val_patch_metrics,
        test_patch_metrics,
        val_patient_metrics,
        test_patient_metrics,
        val_mil_metrics,
        test_mil_metrics,
        cfg,
    )

    print(f"\nVirchow pipeline completed in {time.time() - total_start:.2f}s")
    print("Outputs saved to:", cfg.OUTPUT_DIR)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()
```
