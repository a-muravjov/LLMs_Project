import os
import json
import random

# === CONFIG ===
DATASET_DIR = "datasets"
LANGUAGES = ["EN", "CZ", "RU", "VNM"]  # languages to merge
OUTPUT_DIR = os.path.join(DATASET_DIR, "MULTI")

# === Ensure output folder exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(path):
    """Safely load a JSON file."""
    if not os.path.exists(path):
        print(f"⚠️  Skipping missing file: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_splits(langs, split_name):
    """Merge a given split (train/val/test) from all languages and shuffle."""
    merged = []
    for lang in langs:
        path = os.path.join(DATASET_DIR, lang, f"{split_name}.json")
        data = load_json(path)
        merged.extend(data)
        print(f"Added {len(data)} samples from {lang}/{split_name}.json")

    random.seed(42)
    random.shuffle(merged)

    print(f"Total {split_name} samples after merge: {len(merged)} (shuffled)")
    return merged

def main():
    print("Merging multilingual datasets...")

    for split in ["train", "val", "test"]:
        merged = merge_splits(LANGUAGES, split)
        out_path = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"Saved {split}.json → {out_path}\n")

    print("All multilingual splits merged successfully!")

if __name__ == "__main__":
    main()