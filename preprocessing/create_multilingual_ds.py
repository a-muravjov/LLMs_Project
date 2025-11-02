import os
import json
import random


DATASET_DIR = "datasets"
LANGUAGES = ["EN", "CZ", "RU", "VNM"]
OUTPUT_DIR = os.path.join(DATASET_DIR, "MULTI")


os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_json(path: str) -> dict:
    """
    Reads JSON file.

    Args:
        path (str): File path.

    Returns:
        dict: Data dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_splits(langs: str, split_name: str) -> list:
    """
    Extends the returning dataset by a given split (train/test/val)
    from one of the languages (RU, VNM, CZ, EN) to create a multilingual
    dataset.

    Args:
        langs (str): All chosen languages (RU, VNM, CZ, EN).
        split_name (str): All splits (train/test/val).

    Returns:
        list: Final merged dictionary.
    """
    merged = []

    for lang in langs:
        path = os.path.join(DATASET_DIR, lang, f"{split_name}.json")
        data = read_json(path)
        merged.extend(data)

    # shuffle the dataset around to introduce randomness
    random.seed(42)
    random.shuffle(merged)

    return merged


def main() -> None:
    """
    Loops through all splits from all languages and merges them into
    one multilingual dataset.
    """
    for split in ["train", "val", "test"]:
        merged = merge_splits(LANGUAGES, split)
        out_path = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
