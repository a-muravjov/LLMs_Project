import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers.utils import logging
import torch

logging.set_verbosity_error()

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LEN = 256


def get_tokenizer(model_name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def preprocess_fn(batch, tok, prompt, max_len=MAX_LEN):
    """Tokenize and mask prompt+label pairs for causal LM training."""
    input_ids_all, attention_all, labels_all = [], [], []

    for text, ids in zip(batch["text"], batch["gold"]):
        label_names = [LABELS[i - 1] for i in ids]
        label_text = json.dumps(label_names, ensure_ascii=False)

        prompt = (f"<s>[INST]You are an emotion tagger. From this fixed \
            set:\n{LABELS}\n. Return a JSON array of one or more labels. \
            JSON only, no extra text.\n\n Text: '{text}'\nLabels: [/INST]")

        p = tok(prompt, add_special_tokens=True)
        t = tok(label_text, add_special_tokens=False)

        inp = p["input_ids"] + t["input_ids"]
        attention = p["attention_mask"] + t["attention_mask"]
        labels = [-100] * len(p["input_ids"]) + t["input_ids"]

        pad_id = tok.pad_token_id
        if len(inp) > max_len:
            inp, attention, labels = inp[:max_len], attention[:max_len], labels[:max_len]
        else:
            pad = max_len - len(inp)
            inp += [pad_id] * pad
            attention += [0] * pad
            labels += [-100] * pad

        input_ids_all.append(inp)
        attention_all.append(attention)
        labels_all.append(labels)

    return {"input_ids": input_ids_all, "attention_mask": attention_all, "labels": labels_all}


def get_lora_model(model_name=MODEL_NAME):
    """Load base model and apply LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    config = LoraConfig(
        init_lora_weights="pissa",
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    return model


def prepare_datasets(prompt, train_path="train.json", val_path="val.json", tok=None):
    """Load and tokenize train/val datasets."""
    with open(train_path) as f:
        train_ds = json.load(f)
    with open(val_path) as f:
        val_ds = json.load(f)

    train_ds = Dataset.from_list(train_ds)
    val_ds = Dataset.from_list(val_ds)

    train_tok = train_ds.map(lambda x: preprocess_fn(x, tok, prompt), batched=True)
    val_tok = val_ds.map(lambda x: preprocess_fn(x, tok, prompt), batched=True)

    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in cols_to_keep])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names if c not in cols_to_keep])

    return train_tok, val_tok


def get_training_args(title_args):
    return TrainingArguments(
        output_dir=title_args,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,
        do_eval=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
    )


def train_lora_tuning(model, tok, train_tok, val_tok, title_args, title_adapter):
    training_args = get_training_args(title_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok
    )

    trainer.train()
    model.save_pretrained(title_adapter)
    return model


def run_lora_tuning(train_path, val_path, title_args, title_adapter, prompt):
    tok = get_tokenizer()
    model = get_lora_model()
    train_tok, val_tok = prepare_datasets(prompt=prompt, train_path=train_path, val_path=val_path, tok=tok)
    trained_model = train_lora_tuning(model, tok, train_tok, val_tok,
                                      title_args, title_adapter)
    print("LoRA training complete and adapter saved.")
    return trained_model


if __name__ == "__main__":
    run_lora_tuning(
        train_path="datasets/EN/train.json",
        val_path="datasets/EN/val.json",
        title_args="./lora_mistral_en",
        title_adapter="./lora_adapter_en"
    )