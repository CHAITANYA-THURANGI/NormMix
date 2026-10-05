import json
from pathlib import Path

# Path to Aksharantar Telugu training data
file_path = Path("data/external/aksharantar/te/tel_train.json")

# Store records here
data = []

# Read JSON Lines file
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line:
            data.append(json.loads(line))

print("Dataset loaded successfully!")
print("Number of records:", len(data))

print("\nFirst 5 records:\n")

for record in data[:5]:
    print(record)