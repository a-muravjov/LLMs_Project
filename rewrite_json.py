import json

labels = {
    "anger": 1, "anticipation": 2, "disgust": 3, "fear": 4, "joy": 5,
    "sadness": 6, "surprise": 7, "trust": 8, "neutral": 9
}

with open("predictions.json", "r") as f:
    data_main = json.load(f)

with open("predictions_neu.json", "r") as f:
    data_neu = json.load(f)

offset = max(map(int, data_main.keys())) + 1 if data_main else 0
data_neu_reindexed = {str(i + offset): v for i, v in enumerate(data_neu.values())}

combined = {**data_main, **data_neu_reindexed}

for entry in combined.values():
    entry["pred"] = [labels[p] for p in entry["pred"] if p in labels]

with open("predictions_all.json", "w") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)
