
Advancing the Diagnosis of Head and Neck Small Round Blue Cell Neoplasms through Artificial Intelligence
----------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Dependencies:
    torch>=2.8.0
    torchvision>=0.23.0
    pandas>=2.2.0
    numpy>=1.26.0
    matplotlib>=3.10.0
    scikit-learn>=1.7.0
    pillow>=11.0.0
    tqdm>=4.67.0
    openpyxl>=3.1.0
    xgboost>=3.0.0
    shap>=0.48.0
    lifelines>=0.30.0
    timm>=1.0.0
    openslide-python>=1.4.0
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
import pandas as pd
import shap
import timm
import torch
import torch.nn as nn
import xgboost as xgb

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
        description="UNI feature extraction, patch-level classification, MIL classification, and patient-level evaluation."
    )

    parser.add_argument("--root_dir", type=str, default="./data", help="Path to the input WSI dataset.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to save outputs.")
    parser.add_argument("--uni_dir", type=str, default="./UNI_weights", help="Path to save/load UNI weights.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--patches_per_slide", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--generate_heatmaps", action="store_true", help="Generate attention heatmaps for test slides.")
    parser.add_argument("--heatmap_stride", type=int, default=448)
    parser.add_argument("--max_heatmap_patches", type=int, default=None)

    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="Hugging Face token. Prefer setting HF_TOKEN as an environment variable.",
    )

    return parser.parse_args()


# =========================
# RUNTIME CONFIG HOLDER
# =========================
class Config:
    def __init__(self, args):
        self.ROOT_DIR = args.root_dir
        self.OUTPUT_DIR = args.output_dir
        self.CACHE_DIR = os.path.join(args.output_dir, "cache")
        self.UNI_DIR = args.uni_dir
        self.PLOTS_DIR = os.path.join(args.output_dir, "plots")
        self.METRICS_DIR = os.path.join(args.output_dir, "metrics")
        self.MODELS_DIR = os.path.join(args.output_dir, "models")
        self.HEATMAP_DIR = os.path.join(args.output_dir, "heatmaps")

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

        self.GENERATE_HEATMAPS = args.generate_heatmaps
        self.HEATMAP_STRIDE = args.heatmap_stride
        self.MAX_HEATMAP_PATCHES = args.max_heatmap_patches

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.TQDM_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        for d in [
            self.OUTPUT_DIR,
            self.CACHE_DIR,
            self.UNI_DIR,
            self.PLOTS_DIR,
            self.METRICS_DIR,
            self.MODELS_DIR,
            self.HEATMAP_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


# =========================
# SEED
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# JSON SAFE
# =========================
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


# =========================
# TRANSFORM
# =========================
def build_transform(patch_size):
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
    ])


# =========================
# UNI DOWNLOAD + LOAD
# =========================
def download_uni_if_needed(cfg):
    ckpt_path = os.path.join(cfg.UNI_DIR, "pytorch_model.bin")

    if os.path.exists(ckpt_path):
        print("UNI weights already found locally.")
        return ckpt_path

    print("\n[1/3] Checking Hugging Face authentication...")
    if cfg.HF_TOKEN is not None and str(cfg.HF_TOKEN).strip():
        login(token=cfg.HF_TOKEN, add_to_git_credential=False)
    else:
        print("No token provided in code.")
        print("Use: export HF_TOKEN='your_token'")
        print("Or authenticate previously with: huggingface-cli login")

    print("\n[2/3] Downloading UNI weights...")
    start = time.time()

    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=cfg.UNI_DIR,
        resume_download=True,
    )

    print(f"\n[3/3] Download completed in {time.time() - start:.2f}s")
    print("Weights:", ckpt_path)
    return ckpt_path


def load_uni(cfg):
    ckpt_path = download_uni_if_needed(cfg)

    print("\nLoading UNI model...")
    start = time.time()

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model = model.to(cfg.DEVICE)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print(f"UNI loaded in {time.time() - start:.2f}s")
    return model


# =========================
# CLASS NAME NORMALIZATION
# =========================
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
# PATIENT / CASE ID HELPERS
# =========================
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


# =========================
# DATA DISCOVERY
# =========================
def find_svs_recursively(class_dir):
    class_dir = Path(class_dir)
    svs_files = [str(p) for p in class_dir.rglob("*.svs") if p.is_file()]
    svs_files.sort()
    return svs_files


def load_data(cfg):
    print("\nLoading dataset recursively using fixed four-class setup...\n")
    start = time.time()

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

            samples.append({
                "path": full,
                "label": CLASS_MAP[target_class],
                "label_name": target_class,
                "original_folder": folder_name,
                "case_id": case_id,
                "patient_id": patient_id,
                "relative_path": rel,
                "id": sample_id,
            })

    distribution = dict(Counter([s["label_name"] for s in samples]))

    print(f"\nDataset loaded in {time.time() - start:.2f}s")
    print(f"Total slides found: {len(samples)}")
    print(f"Class map: {CLASS_MAP}")
    print("Distribution:", distribution)

    if skipped_folders:
        print("\nIgnored folders not matching the four target classes:")
        for sf in skipped_folders:
            print(" -", sf)

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


# =========================
# SLIDE VALIDATION
# =========================
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
            invalid_samples.append({
                "path": s["path"],
                "label": s["label"],
                "label_name": s["label_name"],
                "id": s["id"],
                "error": str(e),
            })

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

    if len(rare_classes) > 0:
        print("\nRare classes detected (<3 slides). Using random split with fixed seed.")
        shuffled = samples[:]
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(cfg.TRAIN_RATIO * n)
        n_val = int(cfg.VAL_RATIO * n)

        train_s = shuffled[:n_train]
        val_s = shuffled[n_train:n_train + n_val]
        test_s = shuffled[n_train + n_val:]
        return train_s, val_s, test_s

    train_s, temp_s = train_test_split(
        samples,
        test_size=(1 - cfg.TRAIN_RATIO),
        stratify=labels,
        random_state=cfg.SEED,
    )

    temp_labels = [s["label"] for s in temp_s]
    val_fraction_of_temp = cfg.VAL_RATIO / (1 - cfg.TRAIN_RATIO)

    val_s, test_s = train_test_split(
        temp_s,
        test_size=(1 - val_fraction_of_temp),
        stratify=temp_labels,
        random_state=cfg.SEED,
    )

    return train_s, val_s, test_s


# =========================
# CLASS WEIGHTS
# =========================
def compute_class_weights_from_samples(train_samples, num_classes, cfg):
    counts = Counter([s["label"] for s in train_samples])
    weights = []
    total = len(train_samples)

    for cls_idx in range(num_classes):
        n_cls = counts.get(cls_idx, 0)
        if n_cls == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * n_cls))

    weights = torch.tensor(weights, dtype=torch.float32)

    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes

    print("\nClass weights calculated on the training set:")
    for i, w in enumerate(weights.tolist()):
        print(f" - {TARGET_CLASSES[i]}: {w:.6f} | n={counts.get(i, 0)}")

    save_json(
        {
            "train_counts": {TARGET_CLASSES[i]: counts.get(i, 0) for i in range(num_classes)},
            "class_weights": {TARGET_CLASSES[i]: float(weights[i].item()) for i in range(num_classes)},
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

            patch = slide.read_region((x, y), 0, (self.cfg.PATCH_SIZE, self.cfg.PATCH_SIZE)).convert("RGB")
            slide.close()

        except Exception as e:
            print(f"\nError reading slide: {s['path']}")
            print(f"Reason: {e}")
            patch = Image.new("RGB", (self.cfg.PATCH_SIZE, self.cfg.PATCH_SIZE), (255, 255, 255))

        return self.transform(patch), s["label"], s["id"]


# =========================
# FEATURE EXTRACTION CACHE
# =========================
def extract_or_load(model, loader, name, cfg):
    f_path = os.path.join(cfg.CACHE_DIR, f"{name}_X.pt")
    l_path = os.path.join(cfg.CACHE_DIR, f"{name}_y.pt")
    id_path = os.path.join(cfg.CACHE_DIR, f"{name}_id.pt")

    if os.path.exists(f_path) and os.path.exists(l_path) and os.path.exists(id_path):
        print(f"\nLoading cached features [{name}]")
        start = time.time()
        X = torch.load(f_path)
        y = torch.load(l_path)
        ids = torch.load(id_path)
        print(f"Cache [{name}] loaded in {time.time() - start:.2f}s")
        return X, y, ids

    print(f"\nExtracting features [{name}]...\n")
    start = time.time()

    feats, labels, ids = [], [], []

    with torch.no_grad():
        for x, y, i in tqdm(loader, desc=f"Extracting {name}", dynamic_ncols=True, bar_format=cfg.TQDM_BAR):
            f = model(x.to(cfg.DEVICE))
            if isinstance(f, (list, tuple)):
                f = f[0]

            feats.append(f.cpu())
            labels.append(y)
            ids.extend(i)

    X = torch.cat(feats)
    y = torch.cat(labels)

    torch.save(X, f_path)
    torch.save(y, l_path)
    torch.save(ids, id_path)

    print(f"Features [{name}] saved in {time.time() - start:.2f}s")
    return X, y, ids


# =========================
# BAG CREATION
# =========================
def create_bags(X, y, ids):
    bags = defaultdict(lambda: {"X": [], "y": None})

    for i, sid in enumerate(ids):
        bags[sid]["X"].append(X[i])
        if bags[sid]["y"] is None:
            bags[sid]["y"] = int(y[i].item())

    final_bags = []
    for sid, content in bags.items():
        final_bags.append((sid, torch.stack(content["X"], dim=0), content["y"]))

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
# METRIC HELPERS
# =========================
def get_ovr_confusion_terms(y_true, y_pred, class_idx):
    y_true_bin = (np.array(y_true) == class_idx).astype(int)
    y_pred_bin = (np.array(y_pred) == class_idx).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    return tp, fn, fp, tn


def bootstrap_ci(values, alpha=0.95):
    values = np.array(values, dtype=np.float64)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return None, None

    low = np.percentile(values, (1 - alpha) / 2 * 100)
    high = np.percentile(values, (1 + alpha) / 2 * 100)

    return float(low), float(high)


def bootstrap_auc_ci_multiclass(y_true, probs, n_classes, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    probs = np.array(probs)

    aucs = []
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_true_b = y_true[idx]
        probs_b = probs[idx]
        y_bin_b = y_bin[idx]

        if len(np.unique(y_true_b)) < 2:
            continue

        try:
            auc_val = roc_auc_score(y_bin_b, probs_b, multi_class="ovr", average="macro")
            aucs.append(auc_val)
        except Exception:
            continue

    if len(aucs) == 0:
        return None, None

    return bootstrap_ci(aucs, alpha=0.95)


def bootstrap_binary_metric_ci(y_true_bin, y_pred_bin, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true_bin = np.array(y_true_bin)
    y_pred_bin = np.array(y_pred_bin)

    vals = []
    n = len(y_true_bin)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true_bin[idx]
        yp = y_pred_bin[idx]

        if len(np.unique(yt)) < 2:
            continue

        try:
            vals.append(metric_fn(yt, yp))
        except Exception:
            continue

    if len(vals) == 0:
        return None, None

    return bootstrap_ci(vals, alpha=0.95)


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
    y_true = np.array(y_true)
    probs = np.array(probs)
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def compute_metrics(y_true, y_pred, probs, class_names, cfg, compute_loss_value=None):
    n_classes = len(class_names)
    labels = list(range(n_classes))

    metrics = {
        "loss": float(compute_loss_value) if compute_loss_value is not None else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    y_bin = label_binarize(y_true, classes=labels)

    try:
        metrics["roc_auc_macro_ovr"] = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
    except Exception:
        metrics["roc_auc_macro_ovr"] = None

    try:
        metrics["roc_auc_weighted_ovr"] = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="weighted"))
    except Exception:
        metrics["roc_auc_weighted_ovr"] = None

    auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
        y_true=y_true,
        probs=probs,
        n_classes=n_classes,
        n_boot=1000,
        seed=cfg.SEED,
    )
    metrics["auc_ci_low"] = auc_ci_low
    metrics["auc_ci_high"] = auc_ci_high

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    per_class = {}
    spec_values = []
    sens_values = []

    for i, cname in enumerate(class_names):
        tp, fn, fp, tn = get_ovr_confusion_terms(y_true, y_pred, i)

        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity_cls = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_cls = (
            2 * precision_cls * recall_cls / (precision_cls + recall_cls)
            if (precision_cls + recall_cls) > 0
            else 0.0
        )

        spec_values.append(specificity_cls)
        sens_values.append(recall_cls)

        try:
            auc_cls = roc_auc_score((np.array(y_true) == i).astype(int), probs[:, i])
        except Exception:
            auc_cls = None

        per_class[cname] = {
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
            "precision": float(precision_cls),
            "recall": float(recall_cls),
            "sensitivity": float(recall_cls),
            "specificity": float(specificity_cls),
            "f1_score": float(f1_cls),
            "roc_auc": float(auc_cls) if auc_cls is not None else None,
            "support": int(np.sum(np.array(y_true) == i)),
        }

    metrics["per_class"] = per_class
    metrics["specificity_macro_ovr"] = float(np.mean(spec_values)) if len(spec_values) > 0 else None
    metrics["sensitivity_macro_ovr"] = float(np.mean(sens_values)) if len(sens_values) > 0 else None
    metrics["present_class_indices"] = sorted(list(set(y_true)))
    metrics["missing_class_indices"] = sorted(list(set(labels) - set(y_true)))

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
    plt.figure(figsize=(7, 6))

    y_bin = label_binarize(y_true, classes=labels)
    plotted = False

    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.4f})")
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
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f"{prefix}_loss.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, f"{prefix}_accuracy.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# TRAIN PATCH
# =========================
def train_patch_model(model, X_train, y_train, X_val, y_val, class_weights, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = os.path.join(cfg.MODELS_DIR, "best_patch_model.pt")

    for e in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train))
        losses, preds_all, true_all = [], [], []

        loop = tqdm(
            range(0, len(X_train), cfg.BATCH_SIZE),
            desc=f"Patch Epoch {e + 1}/{cfg.EPOCHS}",
            dynamic_ncols=True,
            bar_format=cfg.TQDM_BAR,
        )

        for i in loop:
            idx = perm[i:i + cfg.BATCH_SIZE]
            xb = X_train[idx].to(cfg.DEVICE)
            yb = y_train[idx].to(cfg.DEVICE)

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            preds_all.extend(torch.argmax(out, dim=1).detach().cpu().numpy().tolist())
            true_all.extend(yb.detach().cpu().numpy().tolist())

            loop.set_postfix(loss=f"{np.mean(losses):.4f}")

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
    plot_training_curves(history, "patch_training", cfg)
    save_json(history, os.path.join(cfg.METRICS_DIR, "patch_training_history.json"))

    return model, history


# =========================
# EVAL PATCH
# =========================
def evaluate_patch_model(model, X, y, class_names, split, class_weights, cfg):
    model.eval()
    probs_list, preds_list, true_list = [], [], []
    losses = []

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    with torch.no_grad():
        loop = tqdm(
            range(0, len(X), cfg.BATCH_SIZE),
            desc=f"Evaluate Patch [{split}]",
            dynamic_ncols=True,
            bar_format=cfg.TQDM_BAR,
        )

        for i in loop:
            xb = X[i:i + cfg.BATCH_SIZE].to(cfg.DEVICE)
            yb = y[i:i + cfg.BATCH_SIZE].to(cfg.DEVICE)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            probs_list.append(probs)
            preds_list.extend(preds.tolist())
            true_list.extend(yb.cpu().numpy().tolist())

    probs = np.concatenate(probs_list, axis=0)
    preds = np.array(preds_list)
    y_true = np.array(true_list)
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else None

    metrics, cm = compute_metrics(y_true, preds, probs, class_names, cfg, compute_loss_value=mean_loss)
    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_patch_metrics.json"))

    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patch_cm.png"), False, f"{split.upper()} Patch CM")
    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patch_cm_norm.png"), True, f"{split.upper()} Patch CM Normalized")
    plot_roc_curve(y_true, probs, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patch_roc.png"))

    return metrics


# =========================
# TRAIN MIL
# =========================
def train_mil_model(model, train_bags, val_bags, class_weights, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = os.path.join(cfg.MODELS_DIR, "best_mil_model.pt")

    for e in range(cfg.EPOCHS):
        model.train()
        random.shuffle(train_bags)

        losses, preds_all, true_all = [], [], []

        loop = tqdm(
            train_bags,
            desc=f"MIL Epoch {e + 1}/{cfg.EPOCHS}",
            dynamic_ncols=True,
            bar_format=cfg.TQDM_BAR,
        )

        for sid, bag_x, bag_y in loop:
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

            loop.set_postfix(loss=f"{np.mean(losses):.4f}")

        train_loss = float(np.mean(losses))
        train_acc = float(accuracy_score(true_all, preds_all))

        model.eval()
        val_losses, val_preds, val_true = [], [], []

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
    plot_training_curves(history, "mil_training", cfg)
    save_json(history, os.path.join(cfg.METRICS_DIR, "mil_training_history.json"))

    return model, history


# =========================
# EVAL MIL
# =========================
def evaluate_mil_model(model, bags, class_names, split, class_weights, cfg):
    model.eval()
    probs_list, preds_list, true_list = [], [], []
    losses = []

    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(cfg.DEVICE))

    with torch.no_grad():
        loop = tqdm(
            bags,
            desc=f"Evaluate MIL [{split}]",
            dynamic_ncols=True,
            bar_format=cfg.TQDM_BAR,
        )

        for sid, bag_x, bag_y in loop:
            bag_x = bag_x.to(cfg.DEVICE)
            yb = torch.tensor([bag_y], dtype=torch.long, device=cfg.DEVICE)

            out = model(bag_x)
            loss = loss_fn(out.unsqueeze(0), yb)
            losses.append(loss.item())

            probs = torch.softmax(out, dim=0).cpu().numpy()
            pred = int(np.argmax(probs))

            probs_list.append(probs[None, :])
            preds_list.append(pred)
            true_list.append(int(bag_y))

    probs = np.concatenate(probs_list, axis=0)
    preds = np.array(preds_list)
    y_true = np.array(true_list)
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else None

    metrics, cm = compute_metrics(y_true, preds, probs, class_names, cfg, compute_loss_value=mean_loss)
    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_mil_metrics.json"))

    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_mil_cm.png"), False, f"{split.upper()} MIL CM")
    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_mil_cm_norm.png"), True, f"{split.upper()} MIL CM Normalized")
    plot_roc_curve(y_true, probs, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_mil_roc.png"))

    return metrics


# =========================
# PATIENT-LEVEL ANALYSIS
# =========================
def parse_sample_id(sample_id):
    parts = str(sample_id).split("__")

    if len(parts) >= 3:
        class_name = parts[0]
        patient_id = parts[1]
        slide_name = "__".join(parts[2:])
        return class_name, patient_id, slide_name

    if len(parts) == 2:
        return parts[0], parts[1], parts[1]

    return "Unknown", str(sample_id), str(sample_id)


def extract_patient_id(sample_id):
    _, patient_id, _ = parse_sample_id(sample_id)
    return patient_id


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

        if len(set(labels)) > 1:
            final_label = Counter(labels).most_common(1)[0][0]
        else:
            final_label = labels[0]

        patient_ids.append(pid)
        patient_y_true.append(final_label)
        patient_probs.append(avg_probs)

    patient_probs = np.array(patient_probs)
    patient_y_true = np.array(patient_y_true)
    patient_y_pred = np.argmax(patient_probs, axis=1)

    return patient_ids, patient_y_true, patient_y_pred, patient_probs


def compute_patient_level_metrics(patient_y_true, patient_y_pred, patient_probs, class_names, cfg):
    n_classes = len(class_names)

    metrics = {
        "accuracy": float(accuracy_score(patient_y_true, patient_y_pred)),
        "ece": expected_calibration_error(patient_y_true, patient_probs, n_bins=10),
        "brier": multiclass_brier_score(patient_y_true, patient_probs, n_classes),
    }

    y_bin = label_binarize(patient_y_true, classes=list(range(n_classes)))

    try:
        metrics["auc"] = float(roc_auc_score(y_bin, patient_probs, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auc"] = None

    auc_ci_low, auc_ci_high = bootstrap_auc_ci_multiclass(
        patient_y_true,
        patient_probs,
        n_classes=n_classes,
        n_boot=1000,
        seed=cfg.SEED,
    )
    metrics["auc_ci_low"] = auc_ci_low
    metrics["auc_ci_high"] = auc_ci_high

    per_class = {}

    for i, cname in enumerate(class_names):
        tp, fn, fp, tn = get_ovr_confusion_terms(patient_y_true, patient_y_pred, i)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        per_class[cname] = {
            "sensitivity": float(sens),
            "specificity": float(spec),
            "TP": int(tp),
            "FN": int(fn),
            "FP": int(fp),
            "TN": int(tn),
        }

    metrics["per_class"] = per_class

    return metrics


def evaluate_patient_level_from_patch_probs(model, X, y, ids, class_names, split, cfg):
    model.eval()
    probs_list, true_list, id_list = [], [], []

    with torch.no_grad():
        loop = tqdm(
            range(0, len(X), cfg.BATCH_SIZE),
            desc=f"Evaluate Patient-Level from Patch [{split}]",
            dynamic_ncols=True,
            bar_format=cfg.TQDM_BAR,
        )

        for i in loop:
            xb = X[i:i + cfg.BATCH_SIZE].to(cfg.DEVICE)
            yb = y[i:i + cfg.BATCH_SIZE]

            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            probs_list.append(probs)
            true_list.extend(yb.numpy().tolist())
            id_list.extend(ids[i:i + cfg.BATCH_SIZE])

    probs = np.concatenate(probs_list, axis=0)
    y_true = np.array(true_list)

    patient_ids, patient_y_true, patient_y_pred, patient_probs = aggregate_to_patient_level(
        ids=id_list,
        y_true=y_true,
        probs=probs,
    )

    metrics = compute_patient_level_metrics(
        patient_y_true=patient_y_true,
        patient_y_pred=patient_y_pred,
        patient_probs=patient_probs,
        class_names=class_names,
        cfg=cfg,
    )

    cm = confusion_matrix(patient_y_true, patient_y_pred, labels=list(range(len(class_names))))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["n_patients"] = len(patient_ids)

    save_json(metrics, os.path.join(cfg.METRICS_DIR, f"{split}_patient_level_metrics.json"))

    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patient_cm.png"), False, f"{split.upper()} Patient CM")
    plot_confusion_matrix(cm, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patient_cm_norm.png"), True, f"{split.upper()} Patient CM Normalized")
    plot_roc_curve(patient_y_true, patient_probs, class_names, os.path.join(cfg.PLOTS_DIR, f"{split}_patient_roc.png"))

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
    val_mil_metrics,
    test_mil_metrics,
    val_patient_metrics,
    test_patient_metrics,
    cfg,
):
    summary = {
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
        "val_mil_metrics": val_mil_metrics,
        "test_mil_metrics": test_mil_metrics,
        "val_patient_metrics": val_patient_metrics,
        "test_patient_metrics": test_patient_metrics,
    }

    save_json(summary, os.path.join(cfg.OUTPUT_DIR, "final_summary.json"))


# =========================
# MAIN
# =========================
def main():
    args = parse_args()
    cfg = Config(args)

    total_start = time.time()
    set_seed(cfg.SEED)

    print("\nUNI pipeline: four-class classification with patch-level, patient-level, and MIL analysis\n")
    print(f"Device: {cfg.DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = build_transform(cfg.PATCH_SIZE)

    samples, class_map = load_data(cfg)

    if len(samples) == 0:
        raise RuntimeError("No .svs slides were found for the four target classes.")

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

    uni = load_uni(cfg)

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

    Xtr, ytr, idtr = extract_or_load(uni, train_loader, "train", cfg)
    Xv, yv, idv = extract_or_load(uni, val_loader, "val", cfg)
    Xte, yte, idte = extract_or_load(uni, test_loader, "test", cfg)

    class_names = TARGET_CLASSES

    print("\nTraining patch classifier...\n")
    patch_model = PatchClassifier(Xtr.shape[1], len(class_names)).to(cfg.DEVICE)
    patch_model, _ = train_patch_model(
        patch_model,
        Xtr,
        ytr,
        Xv,
        yv,
        class_weights=class_weights,
        cfg=cfg,
    )

    print("\nEvaluating patch classifier...\n")
    val_patch_metrics = evaluate_patch_model(
        patch_model,
        Xv,
        yv,
        class_names,
        split="val",
        class_weights=class_weights,
        cfg=cfg,
    )
    test_patch_metrics = evaluate_patch_model(
        patch_model,
        Xte,
        yte,
        class_names,
        split="test",
        class_weights=class_weights,
        cfg=cfg,
    )

    print("\nEvaluating patient-level predictions from patch probabilities...\n")
    val_patient_metrics = evaluate_patient_level_from_patch_probs(
        patch_model,
        Xv,
        yv,
        idv,
        class_names,
        split="val",
        cfg=cfg,
    )
    test_patient_metrics = evaluate_patient_level_from_patch_probs(
        patch_model,
        Xte,
        yte,
        idte,
        class_names,
        split="test",
        cfg=cfg,
    )

    print("\nCreating bags for MIL...\n")
    train_bags = create_bags(Xtr, ytr, idtr)
    val_bags = create_bags(Xv, yv, idv)
    test_bags = create_bags(Xte, yte, idte)

    print("\nTraining MIL attention model...\n")
    mil_model = MILAttention(Xtr.shape[1], len(class_names)).to(cfg.DEVICE)
    mil_model, _ = train_mil_model(
        mil_model,
        train_bags,
        val_bags,
        class_weights=class_weights,
        cfg=cfg,
    )

    print("\nEvaluating MIL attention model...\n")
    val_mil_metrics = evaluate_mil_model(
        mil_model,
        val_bags,
        class_names,
        split="val",
        class_weights=class_weights,
        cfg=cfg,
    )
    test_mil_metrics = evaluate_mil_model(
        mil_model,
        test_bags,
        class_names,
        split="test",
        class_weights=class_weights,
        cfg=cfg,
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
        val_mil_metrics,
        test_mil_metrics,
        val_patient_metrics,
        test_patient_metrics,
        cfg,
    )

    print(f"\nPipeline completed in {time.time() - total_start:.2f}s")
    print("Outputs saved to:", cfg.OUTPUT_DIR)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
