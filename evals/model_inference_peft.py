import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()

# === CONFIG ===
MODEL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
MODELS_DIR = "models"
DATASETS_DIR = "datasets"
OUTPUT_DIR = "predictions"
BATCH_SIZE = 4

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# === Load base + adapter ===
def load_model(adapter_path):
    tok = AutoTokenizer.from_pretrained(MODEL_BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        device_map="auto",
        torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipe


# === Build prompt ===
def make_prompt(text: str) -> str:
    return (f"<s>[INST]You are an emotion tagger. From this fixed \
            set:\n{LABELS}\n. Return a JSON array of one or more labels. \
            JSON only, no extra text.\n\n Text: '{text}'\nLabels: [/INST]")


# === Extract predicted labels ===
def parse_labels(s: str):
    s = s.split("\nText:")[0].strip()
    m = re.search(r"\[.*?\]", s, flags=re.S)
    pred = []
    if m:
        try:
            arr = json.loads(m.group(0))
            pred = [x.lower() for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    if not pred:
        pred = [p.strip().lower() for p in s.splitlines()[0].split(",")]
    seen = set()
    return [p for p in pred if p in LABELS and not (p in seen or seen.add(p))]



# === Inference loop ===
def run_inference(adapter_name, dataset_lang="EN"):
    adapter_path = os.path.join(MODELS_DIR, adapter_name)
    dataset_path = os.path.join(DATASETS_DIR, dataset_lang, "test.json")

    output_name = adapter_name.replace("adapter_", "predictions_") + ".json"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print(f"Running inference for {adapter_name}")
    print(f"Using dataset: {dataset_path}")
    print(f"Output: {output_path}")

    pipe = load_model(adapter_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = {}
    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start:start + BATCH_SIZE]
        prompts = [make_prompt(x["text"]) for x in batch]
        outs = pipe(
            prompts,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
            batch_size=BATCH_SIZE
        )

        for j, out in enumerate(outs):
            text = batch[j]["text"]
            gold = batch[j]["gold"]
            gen = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            preds = parse_labels(gen)
            # --- Added raw output for debugging ---
            results[start + j] = {"text": text, "gold": gold, "pred": preds, "raw": gen}

        if start % 50 == 0:
            print(f"Processed {start + BATCH_SIZE}/{len(samples)} samples")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Finished {adapter_name}! Saved predictions to {output_path}")



# === MAIN SCRIPT ===
if __name__ == "__main__":
    adapters = [
        d for d in os.listdir(MODELS_DIR)
        if (
            os.path.isdir(os.path.join(MODELS_DIR, d))
            and "_adapter_" in d
            and any(x in d for x in ["lora", "qlora"])
            and any(x in d for x in ["EN", "MULTI", "RU", "CZ", "VNM"])
        )
    ]

    print(f"Found {len(adapters)} adapters to evaluate:")
    for a in adapters:
        print(" -", a)

    for adapter in adapters:
        lang = None
        for possible in ["EN", "MULTI", "RU", "CZ", "VNM"]:
            if possible in adapter:
                lang = possible
                break

        # fallback safety — just skip if language is not found
        if lang is None:
            print(f"Skipping {adapter}, could not detect language.")
            continue

        try:
            run_inference(adapter, dataset_lang=lang)
        except Exception as e:
            print(f"Failed {adapter}: {e}")