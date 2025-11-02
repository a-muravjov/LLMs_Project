import os
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from transformers.utils import logging

logging.set_verbosity_error()


MODELS_DIR = "models"
DATASETS_DIR = "datasets"
OUTPUT_DIR = "predictions"
BATCH_SIZE = 4

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_pipeline(adapter_path: str):
    """
    Loads a pipeline with fine-tuned Mistral-7B-v0.3 model and tokenizer.

    Args:
        adapter_path (str): type of PEFT method.

    Returns:
        Pipeline: The pipeline.
    """
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        device_map="auto",
        torch_dtype="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)

    ppl = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map="auto",
        torch_dtype="auto"
    )
    return ppl


def make_prompt(text: str) -> str:
    """
    Creates an instruction based prompt.

    Args:
        text (str): Text to identify emotion.

    Returns:
        str: Instruction based prompt.
    """
    return (f"<s>[INST]You are an emotion tagger. From this fixed \
            set:\n{LABELS}\n. Return a JSON array of one or more labels. \
            JSON only, no extra text.\n\n Text: '{text}'\nLabels: [/INST]")


def parse_labels(raw_output: str) -> list:
    """
    Cleans the raw output of the model to extract label predictions.
    Handles cases from all prompt engineering methods and PEFT model
    variations.

    Args:
        raw_output (str): Raw output of the model.

    Returns:
        list: List of predictions.
    """
    # the model often separated the predicted labels and new, irrelevant,
    # output by newline characters
    raw_output = raw_output.split("\n", 1)[0].strip()

    # the model often makes up a new text to conduct emotion classification
    # on - to avoid extracting irrelevant label, cut off at first occurence
    # of "Text"
    raw_output = re.split(r"Text:|\]", raw_output)[0].strip()

    # substite various non-word characters with empty string to catch
    # matches better
    raw_output = re.sub(r"\d+[\.\)]\s*", "", raw_output)

    if not raw_output:
        return []

    # instruction and structure based prompting specific parsing
    # parses by finding [], [//]..
    ####
    match = re.search(r"\[.*?\]", raw_output, flags=re.S)
    pred = []

    if match:
        try:
            arr = json.loads(match.group(0))
            pred = [x.lower() for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            cleaned = re.sub(r"[\[\]\"']", "", match.group(0))
            pred = [p.strip().lower() for p in cleaned.split(",")
                    if p.strip() in LABELS]
    ####

    # find all the present labels in the cut down version of the raw output
    if not pred:
        found = re.findall(r"\b(" + "|".join(LABELS) + r")\b", raw_output,
                           flags=re.IGNORECASE)
        pred = [f.lower() for f in found if f.lower() in LABELS]

    seen = set()
    pred = [p for p in pred if p in LABELS and not (p in seen or seen.add(p))]

    return pred


def run_inference(adapter_name: str, dataset_lang: str) -> dict:
    # define the input and adapter paths
    adapter_path = os.path.join(MODELS_DIR, adapter_name)
    dataset_path = os.path.join(DATASETS_DIR, dataset_lang, "test.json")

    output_name = adapter_name.replace("adapter_", "predictions_") + ".json"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    pipe = get_pipeline(adapter_path)

    # load test datasets
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = {}
    # loop through all prompts
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
        # loop though individual outputs for each text example
        for j, out in enumerate(outs):
            text = batch[j]["text"]
            gold = batch[j]["gold"]
            if isinstance(out, list):
                gen = out[0]["generated_text"]
            else:
                gen = out["generated_text"]
            preds = parse_labels(gen)
            # keep the raw text; as the model's output is not consistent
            # it provides better understanding + used for debugging parsing
            # methods
            results[start + j] = {"text": text, "gold": gold, "pred": preds,
                                  "raw": gen}

        if start % 50 == 0:
            print(f"Processed {start + BATCH_SIZE}/{len(samples)} samples")

    # save test dataset, not with predictions
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


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
    for adapter in adapters:
        lang = None
        for possible in ["EN", "MULTI", "RU", "CZ", "VNM"]:
            if possible in adapter:
                lang = possible
                break
        try:
            run_inference(adapter, dataset_lang=lang)
        except Exception as e:
            print(f"Failed {adapter}: {e}")
