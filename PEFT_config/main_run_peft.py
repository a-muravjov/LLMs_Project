import os
from prefix_tuning import run_prefix
from lora_tuning import run_lora_tuning
from qlora_tuning import run_qlora_tuning

# Later you can import: from lora_tuning import train_lora_tuning, etc.

def main():
    '''
    # === Choose dataset ===
    lang = "EN"  # for now, only English
    base_path = os.path.join("datasets", lang)
    train_path = os.path.join(base_path, "train.json")
    val_path = os.path.join(base_path, "val.json")

    # === Run prefix tuning ===
    print(f"🚀 Starting Prefix Tuning for {lang} dataset")
    run_prefix(
        train_path="datasets/EN/train.json",
        val_path="datasets/EN/val.json",
        title_args="models/prefix_tuning_mistral_en",
        title_adapter="models/prefix_tuning_adapter_en"
    )
    print(f"Finished training Prefix Tuning for {lang} dataset!")


    print("Starting LoRA fine-tuning for EN dataset")
    run_lora_tuning(
        train_path="datasets/EN/train.json",
        val_path="datasets/EN/val.json",
        title_args="models/lora_mistral_en",
        title_adapter="models/lora_adapter_en"
    )
    

    lang = ["EN", "MULTI", "RU", "CZ", "VNM"]
    rank_alpha = [(16, 8), (16, 16), (8, 16)]


    for language in lang:
        for rank, alpha in rank_alpha:
            print(f"Starting QLoRA fine-tuning for {language} dataset (rank={rank}, alpha={alpha})")
            run_qlora_tuning(
                prompt=language,
                train_path=f"datasets/{language}/train.json",
                val_path=f"datasets/{language}/val.json",
                title_args=f"models/qlora_mistral_{language}_r{rank}_a{alpha}",
                title_adapter=f"models/qlora_adapter_{language}_r{rank}_a{alpha}",
                rank=rank,
                alpha=alpha
            )

    lang = ["EN", "MULTI", "RU", "CZ", "VNM"]
    rank_alpha = [(16, 8), (16, 16), (8, 16)]


    for language in lang:
        for rank, alpha in rank_alpha:
            print(f"Starting QLoRA fine-tuning for {language} dataset (rank={rank}, alpha={alpha})")
            run_qlora_tuning(
                prompt=language,
                train_path=f"datasets/{language}/train.json",
                val_path=f"datasets/{language}/val.json",
                title_args=f"models/qlora_mistral_{language}_r{rank}_a{alpha}",
                title_adapter=f"models/qlora_adapter_{language}_r{rank}_a{alpha}",
                rank=rank,
                alpha=alpha
            )
    '''
    lang = ["EN", "MULTI", "RU", "CZ", "VNM"]
    for language in lang:
        print(f"Starting Prefix Tuning for {language} dataset")
        run_prefix(
            train_path=f"datasets/{language}/train.json",
            val_path=f"datasets/{language}/val.json",
            title_args=f"models/prefix_tuning_mistral_{language}",
            title_adapter=f"models/prefix_tuning_adapter_{language}"
        )



if __name__ == "__main__":
    main()