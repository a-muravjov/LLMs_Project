import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    jaccard_score,
    f1_score,
    precision_recall_fscore_support,
    hamming_loss
)
from collections import Counter

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS, start=1)}
RESULTS_CSV = "evals/figures/results_summary.csv"


def load_json(path: str) -> json:
    """
    Loads the JSON file from a specified path.

    Args:
        path (str): Path name

    Returns:
        json: JSON file
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_label_indices(labels: list) -> list:
    """
    Converts emotion labels from string to the corresponding
    integer.

    Args:
        labels (list): Predictions in str format

    Returns:
        list: Converted predictions in int format
    """
    ids = []
    for lbl in labels:
        if isinstance(lbl, int):
            ids.append(lbl)
        elif isinstance(lbl, str) and lbl.lower() in LABEL2IDX:
            ids.append(LABEL2IDX[lbl.lower()])
    return [i for i in ids if i is not None]


def evaluate_predictions(data: json) -> dict:
    """
    Computes all metrics (Jaccard Index, Micro F1, Macro F1,
    Precision, Recall, Hamming Loss, F1 per emotion) for a
    given prediction file.

    Args:
        data (json): Prediction JSON file

    Returns:
        dict: Dictionary of all metrics
    """
    n_labels = len(LABELS)
    y_true = np.zeros((len(data), n_labels), dtype=int)
    y_pred = np.zeros((len(data), n_labels), dtype=int)

    for idx, entry in enumerate(data.values()):
        gold = get_label_indices(entry.get("gold", []))
        pred = get_label_indices(entry.get("pred", []))
        for g in gold:
            y_true[idx][g - 1] = 1
        for p in pred:
            y_pred[idx][p - 1] = 1

    jacc = jaccard_score(y_true, y_pred, average="samples")
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )
    hamm = hamming_loss(y_true, y_pred)

    # x,y,z variables not used
    x, y, per_label_f1, z = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)

    return {
        "Jaccard": jacc,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Precision": prec,
        "Recall": rec,
        "Hamming": hamm,
        "PerLabelF1": per_label_f1,
    }


def identify_metadata(path_str: str) -> tuple:
    """
    Identifies the model and language from the path name of
    the prediction dataset.

    Args:
        path_str (str): Path name

    Returns:
        tuple: The model and language.
    """
    path_str = path_str.lower()

    if "qlora" in path_str:
        model = "QLoRA"
    elif "lora" in path_str and "qlora" not in path_str:
        model = "LoRA"
    elif "few" in path_str:
        model = "Few-shot Prompting"
    elif "zero" in path_str:
        model = "Zero-shot Prompting"
    elif "structure" in path_str:
        model = "Structure Prompting"
    elif "instruction" in path_str:
        model = "Instruction-based Prompting"
    else:
        model = "Unknown"

    lang_map = {
        "MULTI": ["multi"],
        "VNM": ["vnm", "vn"],
        "EN": ["en"],
        "CZ": ["cz", "cs"],
        "RU": ["ru", "rus"]
    }
    lang = next((k for k, v in lang_map.items()
                 if any(a in path_str for a in v)), "Unknown")

    return model, lang


def summarize_results() -> pd.DataFrame:
    """
    Loads all of the prediction datasets, computes metrics,
    and puts everything into a single dataframe.

    Returns:
        pd.DataFrame: Dataframe with all of the metrics per model
    """
    all_paths = list(Path("predictions").rglob("*.json"))
    records = []

    for path in all_paths:
        data = load_json(path)
        print(f"Loaded {len(data)} samples from {path}")
        metrics = evaluate_predictions(data)

        model, lang = identify_metadata(str(path))
        row = {
            "Path": str(path),
            "Model": model,
            "Language": lang,
            "Samples": len(data),
            **{k: v for k, v in metrics.items() if k != "PerLabelF1"},
        }

        for label, f1 in zip(LABELS, metrics["PerLabelF1"]):
            row[f"F1_{label}"] = f1

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved {len(df)} results to {RESULTS_CSV}")
    return df


def analyze_dataset(path: str) -> None:
    """
    Prints label frequency per dataset, specifically single label
    versus multi label counts.

    Args:
        path (str): Dataset to analyze
    """
    data = load_json(path)
    standalone_counts = Counter()
    multi_counts = Counter()

    for sample in data:
        gold = sample.get("gold", [])
        if len(gold) == 1:
            standalone_counts[gold[0]] += 1
        else:
            for g in gold:
                multi_counts[g] += 1

    print(f"\nLabel Frequency Summary ({path})\n")
    print(
        f"{'Label':<15}{'Standalone':>15}{'In Multi-label':>20}{'Total':>15}")
    print("-" * 70)
    for i, label in enumerate(LABELS, start=1):
        standalone = standalone_counts[i]
        multi = multi_counts[i]
        total = standalone + multi
        print(f"{label:<15}{standalone:>15}{multi:>20}{total:>15}")
    print("-" * 70)
    print(f"Total samples: {len(data)}")


def plot_per_emotion_heatmap(df: pd.DataFrame) -> None:
    """
    Plots the heatmap that compares F1 scores between all languages and
    emotions, averaged by the models.

    Args:
        df (pd.DataFrame): Dataset with the metrics.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    emotion_cols = [c for c in df.columns if c.startswith("F1_")]
    melted = df.melt(
        id_vars=["Language", "Model"],
        value_vars=emotion_cols,
        var_name="Emotion",
        value_name="F1"
    )
    melted["Emotion"] = melted["Emotion"].str.replace("F1_", "")
    avg_emotion = melted.groupby(
        ["Language", "Emotion"])["F1"].mean().unstack()

    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_emotion, annot=True, cmap="YlGnBu",
                fmt=".2f", linewidths=0.5)
    plt.title("Per-Emotion F1 Across Languages (Averaged over Models)",
              fontsize=14)
    plt.ylabel("Language")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig("evals/figures/heatmap.png")
    plt.show()


def plot_metrics_by_model(df: pd.DataFrame) -> None:
    """
    Plots a separate graph for each metric, which includes a comparison
    of performance between all models and languages.

    Args:
        df (pd.DataFrame): Dataset with the metrics.
    """
    sns.set(style="whitegrid", font_scale=1.2)
    avg_lang = df.groupby(["Language", "Model"], as_index=False)[
        ["Jaccard", "Micro F1", "Macro F1", "Precision", "Recall", "Hamming"]
    ].mean()

    model_order = [
        "Zero-shot Prompting",
        "Few-shot Prompting",
        "Structure Prompting",
        "Instruction-based Prompting",
        "LoRA",
        "QLoRA"
    ]

    metrics = {
        "Jaccard": "Jaccard Score",
        "Micro F1": "Micro F1",
        "Macro F1": "Macro F1",
        "Precision": "Precision",
        "Recall": "Recall",
        "Hamming": "Hamming Loss"
    }

    for metric, label in metrics.items():
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=avg_lang,
            y="Language",
            x=metric,
            hue="Model",
            hue_order=[m for m in model_order
                       if m in avg_lang["Model"].unique()],
            palette="Paired",
            orient="h"
        )
        plt.title(f"Average {label} per Language and Model", fontsize=14)
        plt.xlabel(label)
        plt.ylabel("Language")
        plt.xlim(0, 1)
        if "Hamming" in metric:
            plt.xlim(1, 0)
        plt.legend(title="Model", fontsize=9, title_fontsize=10, loc="best",
                   frameon=True)
        plt.tight_layout()
        plt.savefig(f"evals/figures/{metric}.png")
        plt.show()


def plot_lora_vs_qlora(df: pd.DataFrame, metric: str) -> None:
    """
    Plots a comparison of all the configurations of LoRA and QLoRA
    for a chosen metric per each language.

    Args:
        df (pd.DataFrame): Dataset with the metrics.
        metric (str): Specified metric.
    """
    qlora_df = df[df["Model"] == "QLoRA"].copy()
    qlora_df["Rank"] = qlora_df["Path"].apply(
        lambda x: int(re.search(
            r"r(\d+)", x).group(1)) if re.search(r"r(\d+)", x) else None
    )
    qlora_df["Alpha"] = qlora_df["Path"].apply(
        lambda x: int(re.search(
            r"a(\d+)", x).group(1)) if re.search(r"a(\d+)", x) else None
    )
    qlora_df["Variant"] = qlora_df.apply(
        lambda row: f"QLoRA (r{int(row['Rank'])}_a{int(row['Alpha'])})"
        if pd.notnull(row['Rank']) and pd.notnull(row['Alpha'])
        else "QLoRA (unknown)", axis=1
    )

    lora_df = df[df["Model"] == "LoRA"].copy()
    lora_df["Variant"] = "LoRA"

    combined = pd.concat([qlora_df, lora_df], ignore_index=True)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=combined,
        y="Language",
        x=metric,
        hue="Variant",
        palette="Set2",
        orient="h",
        edgecolor="black"
    )
    plt.title(
        f"LoRA and QLoRA ({metric}) Comparison Across Languages and Variants",
        fontsize=14)
    plt.xlim(0, 1)
    plt.xlabel(f"{metric} Score")
    plt.ylabel("Language")
    plt.legend(title="Model Variant", loc="lower right", fontsize=9,
               title_fontsize=10)
    plt.tight_layout()
    plt.savefig("evals/figures/lora_vs_qlora.png")
    plt.show()
