# Large Language Models (WBAI068-05)

### Authors: Arseni Muravjov, Daniella Alves, Lam Anh Nguyen

```
├── datasets/
│   ├── CZ/
│   │   ├── train.json         # all language datasets contains the same split
│   │   ├── val.json
│   │   └── test.json
│   ├── EN/
│   ├── RU/
│   ├── VNM/
│   └── MULTI/
│
├── evals/
│   ├── evaluate_all.py   
│   ├── run_evaluation.py      # evaluates performance of all model variations
│   ├── figures/               # all relevant figures
├── mistral_peft.py
|   ├── mistral_finetuned.py   # runs predictions for finetuned models
├── mistral_prompting.py      
│   ├── mistral_model.py       # defines functions for running different
│   ├── run_prompting.py       # prompt engineering techniques
│
├── peft_config/
│   ├── lora_tuning.py         # LoRA hyperparameter tuning
│   ├── qlora_tuning.py        # QLoRA hyperparameter tuning
│   ├── main_run_peft.py       # finetunes model with all PEFT configurations
│
├── predictions/
│   ├── few_shot/
│   ├── instruction-based/
│   ├── structure-based/
│   ├── zero_shot/
|   ├── lora_predictions_EN.json
│   ├── ...
|   ├── lora_predictions_CZ.json
│   ├── qlora_predictions_*.json
│   └── ...
│
├── preprocessing/
│   ├── create_multilingual_ds.py # creates a multilingual dataset
│   └── tsv_to_json_split.py      # converts TSV data & splits into train/val/test
│
├── XED/ 
│   ├── cs-projections.tsv       # original TSV datasets containing texts
│   ├── ...                      # to classify
│
├── .gitignore
├── requirements.txt
└── README.md
```


This repository was created for the course project **Cross-Lingual Emotion**
**Classification and Detection Using Large Language Models**. We use multiple
**Mistral-7B-v0.3** variations in order to achieve the best performance on 
chosen subsets of the XED multilingual dataset, which can be found using the
link https://github.com/Helsinki-NLP/XED. The current code implements 2 model
variations: prompt engineering, specifically *zero-shot, few-shot,*
*structured based, and instruction based prompting* and fine-tuning,
specifically using PEFT methods *LoRA and QLoRA*. 

The model performances were compared using the English annotated dataset,
and three projection datasets: Czech, Russian and Vietnamese.

The best results were obtained when using the QLoRA PEFT method, across almost
all datasets. The highest Micro F1 was received with the English dataset,
rank=16 and alpha=8. It is worth mentioning that the current version fails
to explore more suitable combinations and values for rank and alpha, especially
given the varying size of chosen language datasets.

## How to run the project

To preprocess the data, files in preprocessing folder were used and duplicates
removed. If you want to run the predictions yourself, you can skip this step.

To run the model using various prompt engineering methods, run
mistral_prompting/run_prompting.py. Predictions for each language will be found
in predictions/*{insert prompt engineering technique}*. 

To fine-tune and train, run peft_config/main_run_peft.py. This will create
a models folder, which will contain the varying adapters of individual
language and rank & alpha configurations. Using the newly generated adapters,
enter mistral_peft/mistral_finetuned.py to predict. There should be no need
to rewrite any paths. Predictions will be found in
predictions/*{PEFTmethod_predictions_language_rank_alpha.json}*.

Finally, to evaluate, run evals/evaluate_all.py to compare 5 metrics (Jaccard,
Micro/Macro F1, Precision, Recall, Hamming Loss) of all model variations.