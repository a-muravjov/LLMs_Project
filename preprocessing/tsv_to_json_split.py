import os
import json
import csv
from sklearn.model_selection import train_test_split

# === Config ===
XED_DIR = "XED"
OUTPUT_BASE = "datasets"

# Define language mapping: TSV → output folder name
LANG_MAP = {
    "en-annotated.tsv": "EN",
    "cs-projections.tsv": "CZ",
    "ru-projections.tsv": "RU",
    "vi-projections.tsv": "VNM"
}


def parse_tsv(path):
    """Read TSV file and return list of dicts {text, labels, pred}."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            labels = [int(x.strip()) for x in row[1].split(",") if x.strip().isdigit()]
            samples.append({"text": text, "gold": labels})  # dummy pred
    return samples


def split_and_save(lang, samples):
    """Split into train/val/test and save under datasets/{lang}/."""
    train, temp = train_test_split(samples, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    out_dir = os.path.join(OUTPUT_BASE, lang)
    os.makedirs(out_dir, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        print(f"Saved {name}.json → {out_path} ({len(split)} samples)")


def main():
    # --- Parse and split all other languages ---
    for fname, lang in LANG_MAP.items():
        path = os.path.join(XED_DIR, fname)
        if not os.path.exists(path):
            continue
        print(f"Processing {lang} from {fname}")
        samples = parse_tsv(path)
        split_and_save(lang, samples)

    print("Doneee")


if __name__ == "__main__":
    main()