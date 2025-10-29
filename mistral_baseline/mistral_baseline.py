import csv
import json
import re
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]
BATCH_SIZE = 8


def get_pipeline():
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ppl = pipeline("text-generation",
                   model="mistralai/Mistral-7B-Instruct-v0.3",
                   tokenizer=tok,
                   device_map="auto", torch_dtype="auto")

    return ppl, tok


def read_tsv(path: str) -> list:
    with open(path, "r") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if len(r) >= 2]

    return rows

def read_json(path: str) -> list:
    """Reads dataset JSON (with 'text' and 'gold' fields)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [(d["text"], ",".join(map(str, d["gold"]))) for d in data if "text" in d and "gold" in d]
    return rows


def make_prompt(text: str, type_of_prompt: str) -> str:
    if type_of_prompt == "zero-shot":
        return (f"Classify the text into anger, anticipation, disgust, \
                fear, joy, sadness, surprise or trust.\nText: '{text}'\n \
                Emotion: ")
    elif type_of_prompt == "few-shot":
        return (f"Classify the text into anger, anticipation, disgust, \
                fear, joy, sadness, surprise or trust. Here are some examples:\n \
                1. I'm trapped . // fear\n \
                2. If I see you in the stands , it make me feel better . // joy\n \
                3. I'm really sorry . // sadness\n \
                4. Help me ! // trust, surpise, fear\n \
                5. Because I want to . // trust, anger, anticipation, disgust\n \
                \nText: {text} //\n \
                Emotion: ")
    elif type_of_prompt == "structure-based":
        return (f"You are an emotion tagger. From this fixed \
                set:\n{LABELS}\n. Return a JSON array of one or more labels. \
                JSON only, no extra text.\n\n Text: '{text}'\nLabels: ")
    elif type_of_prompt == "instruction-based":
        return (f"<s>[INST]You are an emotion tagger. From this fixed \
                set:\n{LABELS}\n. Return a JSON array of one or more labels. \
                JSON only, no extra text.\n\n Text: '{text}'\nLabels: [/INST]")


def parse_labels(s: str) -> list:
    s = s.split("\nText:")[0].strip()
    if not s:
        return []
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

def include_all_valid_predictions(data: dict) -> dict:
    for _, entry in data.items():
        if not entry["pred"]:
            found_labels = re.findall(r"\b(" + "|".join(LABELS) + r")\b", entry["raw"], flags=re.IGNORECASE)
            cleaned = [label.lower() for label in found_labels if label.lower() in LABELS]
            if cleaned:
                entry["pred"] = list(dict.fromkeys(cleaned))
    return data


def make_predictions(rows: list, prompts: list, ppl, tok) -> dict:
    results = {}
    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start:start+BATCH_SIZE]
        outs = ppl(
            batch_prompts,
            max_new_tokens=32,
            do_sample=True, temperature=0.7, top_p=0.9,
            return_full_text=False,
            batch_size=BATCH_SIZE)

        for j, out in enumerate(outs):
            idx = start + j
            text, gold = rows[idx][0], rows[idx][1]
            gold_ids = [int(x.strip()) for x in gold.split(",") if x.strip()]
            if isinstance(out, list):
                gen = out[0]["generated_text"]
            else:
                gen = out["generated_text"]
            preds = parse_labels(gen)
            results[idx] = {"text": text, "gold": gold_ids, "pred": preds,
                            "raw": gen}

    results = include_all_valid_predictions(results)
    return results


def save_preds(results: dict, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run(inp_path: str, out_path: str, prompt: str) -> None:
    ppl, tok = get_pipeline()
    rows = read_json(inp_path)

    prompts = [make_prompt(r[0], prompt) for r in rows]
    results = make_predictions(rows, prompts, ppl, tok)
    save_preds(results, out_path)
