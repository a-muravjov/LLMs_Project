from sklearn.metrics import jaccard_score, f1_score, precision_recall_fscore_support, hamming_loss
import json
import numpy as np

with open("predictions_all.json", "r") as f:
    data = json.load(f)

labels = {
    "anger": 1, "anticipation": 2, "disgust": 3, "fear": 4, "joy": 5,
    "sadness": 6, "surprise": 7, "trust": 8, "neutral": 9
}

y_true = np.zeros((len(data), len(labels)), dtype=int)
y_pred = np.zeros((len(data), len(labels)), dtype=int)

print(y_true.shape)

for idx, value in enumerate(data.values()):
    ideal = data[str(idx)]["gold"]
    pred = data[str(idx)]["pred"]
    for i in ideal:
        y_true[idx][i-1] = 1
    for i in pred:
        y_pred[idx][i-1] = 1

print("PRIMARY:")
print(f"Jaccard: {jaccard_score(y_true, y_pred, average='samples')}")
print(f"Micro F1: {f1_score(y_true, y_pred, average='micro')}\n")
print("SECONDARY:")
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro')}")
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='samples')
print("Precision:", prec)
print("Recall:", rec)
hl = hamming_loss(y_true, y_pred)
print("Hamming Loss:", hl)