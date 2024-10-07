import os
import json
from tqdm import tqdm

dirname = os.path.dirname(__file__)
dataset_path = os.path.join(dirname, "../data/MEDI2")

files = os.listdir(dataset_path)

data_files = []
for file in files:
    if file.endswith(".jsonl") and not file.startswith("task"):
        data_files.append(file)
        
all_lines = []
for file in tqdm(data_files):
    with open(os.path.join(dataset_path, file), "r") as f:
        lines = f.readlines()
        all_lines.extend([json.loads(line) for line in lines])
        
queries = []
documents = []

for i in tqdm(range(len(all_lines))):
    queries.append(all_lines[i]["query"][1])
    documents.append(all_lines[i]["pos"][0][1])
    documents.append(all_lines[i]["neg"][0][1])
    
documents = list(set(documents))
queries = list(set(queries))   

with open(os.path.join(dirname, "../data/medi2_documents-t.json"), "w") as f:
    json.dump(documents, f)
