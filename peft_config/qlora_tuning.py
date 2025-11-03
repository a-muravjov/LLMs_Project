import torch
import json
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from transformers.utils import logging

logging.set_verbosity_error()

LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
          "sadness", "surprise", "trust"]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LEN = 256


def get_tokenizer(model_name: str = MODEL_NAME):
    """
    Loads tokenizer.

    Args:
        model_name (str): Mistral-7B-v.03.

    Returns:
        Tokenizer: tokenizer.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def preprocess_fn(batch, tok, max_len: int = MAX_LEN) -> dict:
    """
    Tokenize and mask prompt+label pairs for causal LM training.

    Args:
        batch (Dataset): Train and validation sets.
        tok (Tokenizer): Tokenizer.
        max_len (int): Maximum length of tokens. Defaults to MAX_LEN.

    Returns:
        dict: Dictionary of the text, attention mask and label keys, and
        values containing lists of tokenized all aforementioned.
    """
    input_ids_all, attention_all, labels_all = [], [], []

    for text, ids in zip(batch["text"], batch["gold"]):
        label_names = [LABELS[i - 1] for i in ids]  # from 1–9 to 0–8
        label_text = json.dumps(label_names, ensure_ascii=False)

        prompt = (f"<s>[INST]You are an emotion tagger. From this fixed \
            set:\n{LABELS}\n. Return a JSON array of one or more labels. \
            JSON only, no extra text.\n\n Text: '{text}'\nLabels: [/INST]")

        # tokenize both & set add_special_tokens=True, as the prompt
        # uses special model readable tokens
        p = tok(prompt, add_special_tokens=True)
        t = tok(label_text, add_special_tokens=False)

        # Hugging Face standard convention to ignore tokens when computing loss
        ignore_tokens = -100

        # create tokenized input (prompt + label)
        inp = p["input_ids"] + t["input_ids"]
        # set attention - all tokens are paid attention to
        attention = p["attention_mask"] + t["attention_mask"]
        # allows the model to ignore the prompt tokens - penalizes
        # for predicting target tokens
        labels = [ignore_tokens] * len(p["input_ids"]) + t["input_ids"]

        pad_id = tok.pad_token_id
        # pad if the tokenized input is smaller than 256, otherwise cut
        if len(inp) > max_len:
            inp, attention, labels = (inp[:max_len], attention[:max_len],
                                      labels[:max_len])
        else:
            pad = max_len - len(inp)
            inp += [pad_id] * pad
            attention += [0] * pad
            labels += [ignore_tokens] * pad

        input_ids_all.append(inp)
        attention_all.append(attention)
        labels_all.append(labels)

    return {"input_ids": input_ids_all, "attention_mask": attention_all,
            "labels": labels_all}


def get_qlora_model(rank: int, alpha: int, model_name: str = MODEL_NAME):
    """
    Load base model in 4-bit and apply LoRA for QLoRA fine-tuning.

    Args:
        rank (int): rank value.
        alpha (int): alpha value.
        model_name (str): Defaults to Mistral-7B-v0.3.

    Returns:
        PeftModel: Fine-tuned model.
    """
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=q_config,
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

    return model


def prepare_datasets(tok, train_path: str, val_path: str) -> tuple:
    """
    Load and tokenize train/val datasets.

    Args:
        tok (Tokenizer): Tokenizer.
        train_path (str): Train dataset file path.
        val_path (str): Validation dataset file path.

    Returns:
        tuple: Tuple of Datasets.
    """
    with open(train_path) as f:
        train_ds = json.load(f)
    with open(val_path) as f:
        val_ds = json.load(f)

    train_ds = Dataset.from_list(train_ds)
    val_ds = Dataset.from_list(val_ds)

    # tokenize and mask the train and val datasets
    train_tok = train_ds.map(lambda x: preprocess_fn(x, tok), batched=True)
    val_tok = val_ds.map(lambda x: preprocess_fn(x, tok), batched=True)

    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_tok = train_tok.remove_columns([c for c in train_tok.column_names
                                          if c not in cols_to_keep])
    val_tok = val_tok.remove_columns([c for c in val_tok.column_names
                                      if c not in cols_to_keep])

    return train_tok, val_tok


def get_training_args(title_args: str):
    """
    Obtain Training Arguments.

    Args:
        title_args (str): Name to save the file as.

    Returns:
        TrainingArguments: Traning Arguments.
    """
    return TrainingArguments(
        output_dir=title_args,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        do_eval=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
    )


def train_qlora_tuning(model, tok, train_tok, val_tok,
                       title_args: str, title_adapter: str):
    """
    Trains model with QLoRA method.

    Args:
        model (PeftModel): Mistral with LoRA adapter.
        tok (Tokenizer): Tokenizer.
        train_tok (Dataset): Train dataset.
        val_tok (Dataset): Validation dataset.
        title_args (str): Name to save model as.
        title_adapter (str): Name to save adapter as.

    Returns:
        PeftModel: Trained fine-tuned model on chosen data.
    """
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


def run_qlora_tuning(train_path: str, val_path: str, title_args: str,
                     title_adapter: str, rank: int, alpha: int):
    """
    Runs all functions regarding fine-tuning and training model
    using QLoRA.

    Args:
        train_path (str): Training dataset file path.
        val_path (str): Validation dataset file path.
        title_args (str): Name to save model as.
        title_adapter (str): Name to save adapter as.

    Returns:
        PeftModel: Trained and fine-tuned model.
    """
    tok = get_tokenizer()
    model = get_qlora_model(rank, alpha)
    train_tok, val_tok = prepare_datasets(tok, train_path, val_path)
    trained_model = train_qlora_tuning(model, tok, train_tok, val_tok,
                                       title_args, title_adapter)
    print("QLoRA training complete and adapter saved.")
    return trained_model
