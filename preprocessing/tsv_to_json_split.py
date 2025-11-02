import os
import json
import csv
from sklearn.model_selection import train_test_split


XED_DIR = "XED"
OUTPUT_BASE = "datasets"
LANG_MAP = {
    "en-annotated.tsv": "EN",
    "cs-projections.tsv": "CZ",
    "ru-projections.tsv": "RU",
    "vi-projections.tsv": "VNM"
}


def parse_tsv(path: str) -> list:
    """
    Reads a tsv file given by the file path and returns the list
    of dictionaries, each containing keys 'text', 'gold'.

    Args:
        path (str): File path.

    Returns:
        list: List of dictionaries containing the text, its corresponding
        emotion label.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            labels = [int(x.strip()) for x in row[1].split(",")
                      if x.strip().isdigit()]
            samples.append({"text": text, "gold": labels})
    return samples


def split_and_save(lang: str, samples: list) -> None:
    """
    Splits into train/val/test and saves under path 'datasets/{lang}/...'

    Args:
        lang (str): Chosen language (RU, VNM, CZ, EN).
        samples (list): List of dictionaries containing the text,
        its corresponding emotion label.
    """
    train, temp = train_test_split(samples, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    out_dir = os.path.join(OUTPUT_BASE, lang)
    os.makedirs(out_dir, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Loops through all langaguges and their corresponding tsv files
    and creates train/test/val splits and saves them as JSON files.
    """
    for fname, lang in LANG_MAP.items():
        path = os.path.join(XED_DIR, fname)
        if not os.path.exists(path):
            continue
        samples = parse_tsv(path)
        split_and_save(lang, samples)


if __name__ == "__main__":
    main()
