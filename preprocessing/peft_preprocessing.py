import json
from sklearn.model_selection import train_test_split

with open("predictions_all.json") as f:
    data = json.load(f)

samples = [{"text": v["text"], "labels": v["gold"]} for v in data.values()]

train, temp = train_test_split(samples, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(len(train), len(val), len(test))

for name, split in [("train", train), ("val", val), ("test", test)]:
    with open(f"{name}.json", "w") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)