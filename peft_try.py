import csv, json, re
from transformers import pipeline, AutoTokenizer, TrainingArguments, Trainer
from transformers.utils import logging
from peft import get_peft_model, PrefixTuningConfig
from transformers import AutoModelForCausalLM
from datasets import Dataset, DatasetDict, load_dataset
import numpy as np

logging.set_verbosity_error()

LABELS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","neutral"]
TSV_PATH = "XED/AnnotatedData/en-annotated.tsv"
BATCH_SIZE = 8
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LEN = 256

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def preprocess(batch):
    input_ids_all, attention_all, labels_all = [], [], []

    for text, ids in zip(batch["text"], batch["labels"]):
        label_names = [LABELS[i - 1] for i in ids] # from 1-9 to 0-8
        label_text = json.dumps(label_names, ensure_ascii=False)

        prompt = (
            "You are an emotion tagger. From this fixed set:\n"
            f"{LABELS}\n"
            "Return a JSON array of one or more labels. JSON only, no extra text.\n\n"
            f'Text: "{text}"\nLabels: '
        )

        p = tok(prompt, add_special_tokens=False)
        t = tok(label_text, add_special_tokens=False)

        inp = p["input_ids"] + t["input_ids"]
        attention = p["attention_mask"] + t["attention_mask"]
        labels = [-100]*len(p["input_ids"]) + t["input_ids"]

        pad_id = tok.pad_token_id
        if len(inp) > MAX_LEN:
            inp, attention, labels = inp[:MAX_LEN], attention[:MAX_LEN], labels[:MAX_LEN]
        else:
            pad = MAX_LEN - len(inp)
            inp += [pad_id]*pad; attention += [0]*pad; labels += [-100]*pad

        input_ids_all.append(inp)
        attention_all.append(attention)
        labels_all.append(labels)

    return {"input_ids": input_ids_all, "attention_mask": attention_all, "labels": labels_all}

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype="auto", device_map="auto")

peft_config = PrefixTuningConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                num_virtual_tokens=20, # prefix length; try 10, 20, 50
                encoder_hidden_size=model.config.hidden_size, # usually set automatically
                )

model = get_peft_model(model, peft_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")

with open("train.json") as f:
    train_ds = json.load(f)

with open("val.json") as f:
    val_ds = json.load(f)

train_ds = Dataset.from_list(train_ds)
val_ds = Dataset.from_list(val_ds)

train_tok = train_ds.map(preprocess, batched=True)
val_tok = val_ds.map(preprocess, batched=True)

cols_to_keep = ["input_ids", "attention_mask", "labels"]
train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in cols_to_keep])
val_tok = val_tok.remove_columns([c for c in val_tok.column_names   if c not in cols_to_keep])

training_args = TrainingArguments(
  output_dir="./prefix_tuning_mistral",
  num_train_epochs=2,
  per_device_train_batch_size=8,
  per_device_eval_batch_size=8,
  learning_rate=2e-4,
  do_eval=True,
  logging_steps=50,
  eval_strategy="epoch",
  save_total_limit=2,
  remove_unused_columns=False,
  fp16=True, # enable if you have a GPU supporting mixed precision
)


trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_tok,
  eval_dataset=val_tok,
  tokenizer=tok
)

trainer.train()

model.save_pretrained("./prefix_tuning_adapter")