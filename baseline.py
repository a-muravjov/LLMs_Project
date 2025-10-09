import csv, json, re
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

LABELS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","neutral"]
TSV_PATH = "XED/AnnotatedData/en-annotated.tsv"
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

with open(TSV_PATH, "r") as f:
    rows = [r for r in csv.reader(f, delimiter="\t") if len(r) >= 2]

def make_prompt(text: str) -> str:
    return (
        "You are an emotion tagger. From this fixed set:\n"
        f"{LABELS}\n"
        "Return a JSON array of one or more labels. JSON only, no extra text.\n\n"
        f'Text: "{text}"\nLabels: '
    )

prompts = [make_prompt(r[0]) for r in rows]

results = {}
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

for start in range(0, len(prompts), BATCH_SIZE):
    batch_prompts = prompts[start:start+BATCH_SIZE]
    outs = pipe(
        batch_prompts,
        max_new_tokens=32,
        do_sample=True, temperature=0.7, top_p=0.9,
        return_full_text=False,
        batch_size=BATCH_SIZE,
    )
    for j, out in enumerate(outs):
        idx = start + j
        text, gold = rows[idx][0], rows[idx][1]
        gold_ids = [int(x.strip()) for x in gold.split(",") if x.strip()]
        gen = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
        preds = parse_labels(gen)
        results[idx] = {"text": text, "gold": gold_ids, "pred": preds}
    if start % (BATCH_SIZE*20) == 0:
        break

with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
