import json
import re
from transformers import pipeline, AutoTokenizer
from transformers.utils import logging

logging.set_verbosity_error()

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]
BATCH_SIZE = 8


def get_pipeline():
    """
    Loads a pipeline with chosen Mistral-7B-v0.3 model and tokenizer.

    Returns:
        Pipeline: The pipeline.
    """
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    # ensures that EOS token is added
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ppl = pipeline("text-generation",
                   model="mistralai/Mistral-7B-Instruct-v0.3",
                   tokenizer=tok,
                   device_map="auto", torch_dtype="auto")

    return ppl


def read_json(path: str) -> list:
    """
    Loads and reads test sets from datasets for prompting.

    Args:
        path (str): Input path.

    Returns:
        list: List of tuples, each containing a text and its corresponding
        emotion.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # list comprehension for extracting values (text, label) from
    # corresponding keys
    rows = [(d["text"], ",".join(map(str, d["gold"]))) for
            d in data if "text" in d and "gold" in d]

    return rows


def make_prompt(text: str, type_of_prompt: str) -> str:
    """
    Creates a prompt for the LLM depending on the task given in type_of_prompt
    parameter. Possible options are 'zero-shot', 'few-shot', 'structure-based',
    'instruction-based'.

    Args:
        text (str): The text for which we are classifying emotion(s).
        type_of_prompt (str): Type of prompt engineering.

    Returns:
        str: Prompt.
    """
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


def make_predictions(rows: list, prompts: list, ppl) -> dict:
    """
    Creates a dictionary with all text examples, corresponding ground truth
    and model predictions.

    Args:
        rows (list): Text examples.
        prompts (list): Task specific prompts containing text examples.
        ppl (Pipeline): The pipeline.

    Returns:
        dict: Resulting dictionary containing model's predictions.
    """
    results = {}
    # loop through all prompts
    for start in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[start:start+BATCH_SIZE]
        outs = ppl(
            batch_prompts,
            max_new_tokens=32,
            do_sample=True, temperature=0.7, top_p=0.9,
            return_full_text=False,
            batch_size=BATCH_SIZE)
        # loop though individual outputs for each text example
        for j, out in enumerate(outs):
            idx = start + j
            text, gold = rows[idx][0], rows[idx][1]
            gold_ids = [int(x.strip()) for x in gold.split(",") if x.strip()]
            if isinstance(out, list):
                gen = out[0]["generated_text"]
            else:
                gen = out["generated_text"]
            preds = parse_labels(gen)
            # keep the raw text; as the model's output is not consistent
            # it provides better understanding + used for debugging parsing
            # methods
            results[idx] = {"text": text, "gold": gold_ids, "pred": preds,
                            "raw": gen}

    return results


def save_preds(results: dict, out_path: str) -> None:
    """
    Save dictionary into a JSON file.

    Args:
        results (dict): Resulting dictionary containing model's predictions.
        out_path (str): Output path.
    """
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run(inp_path: str, out_path: str, prompt: str) -> None:
    """
    Runs all corresponding functions to emotion classification using
    Mistral-7B-v0.3.

    Args:
        inp_path (str): Input path.
        out_path (str): Output path.
        prompt (str): Chosen prompt technique.
    """
    ppl, tok = get_pipeline()
    rows = read_json(inp_path)

    prompts = [make_prompt(r[0], prompt) for r in rows]
    results = make_predictions(rows, prompts, ppl, tok)
    save_preds(results, out_path)
