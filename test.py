import json

with open("predictions_all.json", "r") as f:
    data = json.load(f)

count = 0
seen = set()
duplicate = set()

for i in data:
    if i["text"] in seen:
        count += 1
        duplicate.add(i["text"])
    else:
        seen.add(i["text"])

print(count)
print(duplicate)