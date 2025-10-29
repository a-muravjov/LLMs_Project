# import json

# with open("datasets/MULTI/train.json", "r") as f:
#     data_train = json.load(f)

# with open("datasets/MULTI/val.json", "r") as f:
#     data_val = json.load(f)

# with open("datasets/MULTI/test.json", "r") as f:
#     data_test = json.load(f)

# combined = data_train + data_val + data_test

# with open("predictions_all.json", "w") as f:
#     json.dump(combined, f, indent=2, ensure_ascii=False)
import json
from datasets import Dataset

with open("datasets/CZ/train.json") as f:
        train_ds = json.load(f)

train_ds = Dataset.from_list(train_ds)
print(train_ds)