from lora_tuning import run_lora_tuning
from qlora_tuning import run_qlora_tuning


def main():
    lang = ["EN", "MULTI", "RU", "CZ", "VNM"]
    rank_alpha = [(16, 8), (16, 16), (8, 16)]

    for language in lang:
        for rank, alpha in rank_alpha:
            print(f"Starting QLoRA fine-tuning for {language} \
                  dataset (rank={rank}, alpha={alpha})")
            run_qlora_tuning(
                train_path=f"datasets/{language}/train.json",
                val_path=f"datasets/{language}/val.json",
                title_args=f"models/qlora_mistral_{language}_r{rank}_a{alpha}",
                title_adapter=f"models/qlora_adapter_{language}_r{rank}_a{alpha}",
                rank=rank,
                alpha=alpha
            )

    for language in lang:
        print(f"Starting LoRA fine-tuning for {language} dataset")
        run_lora_tuning(
            train_path=f"datasets/{language}/train.json",
            val_path=f"datasets/{language}/val.json",
            title_args=f"models/lora_mistral_{language}_r{rank}_a{alpha}",
            title_adapter=f"models/lora_adapter_{language}_r{rank}_a{alpha}"
        )


if __name__ == "__main__":
    main()
