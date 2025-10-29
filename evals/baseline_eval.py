import json
import numpy as np
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_recall_fscore_support,
    hamming_loss
)
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS, start=1)}
RESULTS_CSV = "results_summary.csv"

all_paths = list(Path("predictions").rglob("*.json"))
records = []  # will hold each model/language/prompt’s summary row

for path in all_paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {path}")
    n_labels = len(LABELS)
    y_true = np.zeros((len(data), n_labels), dtype=int)
    y_pred = np.zeros((len(data), n_labels), dtype=int)

    for idx, entry in enumerate(data.values()):
        gold = entry.get("gold", [])
        pred = entry.get("pred", [])

        # Convert gold to numeric
        gold_ids = [i if isinstance(i, int) else LABEL2IDX.get(i, None) for i in gold]
        gold_ids = [i for i in gold_ids if i is not None]

        # Convert predicted string labels → indices
        pred_ids = []
        for p in pred:
            if isinstance(p, int):
                pred_ids.append(p)
            elif isinstance(p, str) and p.lower() in LABEL2IDX:
                pred_ids.append(LABEL2IDX[p.lower()])

        for g in gold_ids:
            y_true[idx][g - 1] = 1
        for p in pred_ids:
            y_pred[idx][p - 1] = 1

    jacc = jaccard_score(y_true, y_pred, average="samples")
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average="samples", zero_division=0)
    hamm = hamming_loss(y_true, y_pred)

    print("\nPRIMARY METRICS")
    print(f"Jaccard: {jacc:.4f}")
    print(f"Micro F1: {micro_f1:.4f}")

    print("\nSECONDARY METRICS")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Hamming Loss: {hamm:.4f}\n")

    per_label_prec, per_label_rec, per_label_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    print("\nPER-EMOTION METRICS")
    for label, p, r, f in zip(LABELS, per_label_prec, per_label_rec, per_label_f1):
        print(f"{label:<15}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")

    print("=" * 100 + "\n")

    fname = path.stem
    prompt_type = "unknown"
    model_type = "unknown"
    lang = "unknown"

    # Heuristics from filename
    path_str = str(path).lower()

    if "qlora" in path_str:
        model_type = "QLoRA"
    elif "lora" in path_str and "qlora" not in path_str:
        model_type = "LoRA"
    elif "prefix" in path_str:
        model_type = "Prefix"

    elif any(x in path_str for x in ["few_shot", "few-shot"]):
        model_type = "Few-shot Prompting"
        prompt_type = "few-shot"
    elif any(x in path_str for x in ["zero_shot", "zero-shot"]):
        model_type = "Zero-shot Prompting"
        prompt_type = "zero-shot"
    elif any(x in path_str for x in ["structure", "structure_based", "structure-based"]):
        model_type = "Structure Prompting"
        prompt_type = "structure-based"
    elif "instruction-based_predictions" in path_str or "instruction-based" in path_str:
        model_type = "Instruction-based Prompting"
        prompt_type = "instruction-based"
    else:
        model_type = "Unknown"
        prompt_type = "Unknown"

    for possible, aliases in {
        "MULTI": ["multi"],
        "VNM": ["vnm", "vn"],
        "EN": ["en"],
        "CZ": ["cz", "cs"],
        "RU": ["ru", "rus"]
    }.items():
        if any(alias in path_str for alias in aliases):
            lang = possible
            break

    row = {
        "Path": str(path),
        "Model": model_type,
        "Prompt": prompt_type,
        "Language": lang,
        "Samples": len(data),
        "Jaccard": jacc,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Precision": prec,
        "Recall": rec,
        "Hamming": hamm,
    }

    # add per-label F1s to columns like "F1_anger", ...
    for label, f in zip(LABELS, per_label_f1):
        row[f"F1_{label}"] = f

    records.append(row)

import json
from collections import Counter

# === CONFIG ===
EN_PATH = "datasets/EN/train.json"  # or val/test.json if you prefer

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]

# === Load dataset ===
with open(EN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} English samples from {EN_PATH}")

# === Counters ===
standalone_counts = Counter()
multi_counts = Counter()

for sample in data:
    gold = sample.get("gold", [])
    if not gold:
        continue

    if len(gold) == 1:
        # single-label case
        standalone_counts[gold[0]] += 1
    else:
        # multi-label case (count all labels involved)
        for g in gold:
            multi_counts[g] += 1

# === Pretty-print results ===
print("Label Frequency Summary (English Dataset)\n")
print(f"{'Label':<15}{'Standalone':>15}{'In Multi-label':>20}{'Total':>15}")
print("-" * 65)

for i, label in enumerate(LABELS, start=1):
    standalone = standalone_counts[i]
    multi = multi_counts[i]
    total = standalone + multi
    print(f"{label:<15}{standalone:>15}{multi:>20}{total:>15}")

print("-" * 65)
print(f"Total samples: {len(data)}")

df = pd.DataFrame(records)
df.to_csv(RESULTS_CSV, index=False)
print(f"Saved {len(df)} results to {RESULTS_CSV}")
print(df.head())

sns.set(style="whitegrid", font_scale=1.2)

emotion_cols = [c for c in df.columns if c.startswith("F1_")]
melted = df.melt(
    id_vars=["Language", "Model"],
    value_vars=emotion_cols,
    var_name="Emotion",
    value_name="F1"
)
melted["Emotion"] = melted["Emotion"].str.replace("F1_", "")

# Average over models per language
avg_emotion = melted.groupby(["Language", "Emotion"])["F1"].mean().unstack()

plt.figure(figsize=(10, 6))
sns.heatmap(avg_emotion, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
plt.title("Per-Emotion F1 Across Languages (Averaged over Models)", fontsize=14)
plt.ylabel("Language")
plt.xlabel("Emotion")
plt.tight_layout()
plt.show()

# --- Aggregate results ---
avg_lang = df.groupby(["Language", "Model"], as_index=False)[
    ["Jaccard", "Micro F1", "Macro F1", "Precision", "Recall", "Hamming"]
].mean()

# --- Model order for consistent display ---
model_order = [
    "Zero-shot Prompting",
    "Few-shot Prompting",
    "Structure Prompting",
    "Instruction-based Prompting",
    "LoRA",
    "QLoRA"
]

# --- Metric list and titles ---
metrics = {
    "Jaccard": "Jaccard Score",
    "Micro F1": "Micro F1",
    "Macro F1": "Macro F1",
    "Precision": "Precision",
    "Recall": "Recall",
    "Hamming": "Hamming Loss"
}

# --- Plot one at a time ---
sns.set(style="whitegrid", font_scale=1.2)

for metric, label in metrics.items():
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=avg_lang,
        y="Language",
        x=metric,
        hue="Model",
        hue_order=[m for m in model_order if m in avg_lang["Model"].unique()],
        palette="Paired",
        orient="h"
    )

    plt.title(f"Average {label} per Language and Model", fontsize=14)
    plt.xlim(0, 1)
    plt.xlabel(label)
    plt.ylabel("Language")

    plt.legend(title="Model", fontsize=9, title_fontsize=10, loc="best", frameon=True)
    plt.tight_layout()
    plt.show()


import re

# --- Extract QLoRA variants ---
qlora_df = df[df["Model"] == "QLoRA"].copy()

# Extract rank/alpha
qlora_df["Rank"] = qlora_df["Path"].apply(
    lambda x: int(re.search(r"r(\d+)", x).group(1)) if re.search(r"r(\d+)", x) else None
)
qlora_df["Alpha"] = qlora_df["Path"].apply(
    lambda x: int(re.search(r"a(\d+)", x).group(1)) if re.search(r"a(\d+)", x) else None
)

qlora_df["Variant"] = qlora_df.apply(
    lambda row: f"QLoRA (r{int(row['Rank'])}_a{int(row['Alpha'])})"
    if pd.notnull(row['Rank']) and pd.notnull(row['Alpha']) else "QLoRA (unknown)",
    axis=1
)

# --- Extract LoRA ---
lora_df = df[df["Model"] == "LoRA"].copy()
lora_df["Variant"] = "LoRA"

# --- Combine ---
combined_df = pd.concat([qlora_df, lora_df], ignore_index=True)

# --- Choose metric to compare ---
metric = "Jaccard"  # or "Micro F1", whichever you prefer

# --- Plot ---
plt.figure(figsize=(9, 6))
sns.barplot(
    data=combined_df,
    y="Language",
    x=metric,
    hue="Variant",
    palette="Set2",
    orient="h",
    edgecolor="black"
)

plt.title(f"LoRA and QLoRA ({metric}) Comparison Across Languages and Variants", fontsize=14)
plt.xlim(0, 1)
plt.xlabel(f"{metric} Score")
plt.ylabel("Language")
plt.legend(title="Model Variant", loc="lower right", fontsize=9, title_fontsize=10)
plt.tight_layout()
plt.show()