import csv, json, re
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

LABELS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","neutral"]
NEU_PATH = "XED/AnnotatedData/neu_en.txt"
BATCH_SIZE = 8

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer=tok,
    device_map="auto",
    torch_dtype="auto"
)

rows = []
with open(NEU_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        label_str, text = parts[0].strip(), parts[1].strip()
        rows.append((text, label_str))

def make_prompt(text: str) -> str:
    return (
        "You are an emotion tagger. From this fixed set:\n"
        f"{LABELS}\n"
        "Return a JSON array of one or more labels. JSON only, no extra text.\n\n"
        f'Text: "{text}"\nLabels: '
    )

def parse_labels(s: str):
    s = s.split("\nText:")[0].strip()
    m = re.search(r"\[.*?\]", s, flags=re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            cand = [str(x).lower().strip() for x in arr]
        except json.JSONDecodeError:
            cand = []
    else:
        # fallback: first line, comma-split
        first = s.splitlines()[0] if s else ""
        cand = [p.strip().lower() for p in first.split(",") if p.strip()]

    seen = set()
    return [p for p in cand if p in LABELS and not (p in seen or seen.add(p))]

prompts = [make_prompt(text) for (text, _gold) in rows]
results = {}

for start in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[start:start + BATCH_SIZE]
    outs = pipe(
        batch_prompts,
        max_new_tokens=32,
        do_sample=False,
        return_full_text=False,
        batch_size=BATCH_SIZE,
    )
    for j, out in enumerate(outs):
        idx = start + j
        text, gold_field = rows[idx]
        gold_ids = [int(x.strip()) for x in gold_field.split(",") if x.strip()]
        gen = out[0]["generated_text"]
        preds = parse_labels(gen)
        results[idx] = {"text": text, "gold": gold_ids, "pred": preds}

    if start % (BATCH_SIZE * 20) == 0:
        print(f"Processed {min(start + BATCH_SIZE, len(prompts))}/{len(prompts)}")

with open("predictions_neu.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Done. Wrote predictions_neu.json")