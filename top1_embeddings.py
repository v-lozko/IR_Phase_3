import os
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = model.to(device)

# Load full training set
msmarco_train = load_dataset("ms_marco", "v2.1", split="train")
os.makedirs("embeddings", exist_ok=True)

all_passages = []
queries = []
neighbors = []

doc_index = 0  # index of each passage in all_passages

for ex in msmarco_train:
    qid = ex["query_id"]
    query = ex["query"]
    passage_texts = ex["passages"]["passage_text"]
    is_selected = ex["passages"]["is_selected"]

    found = False
    for p_text, selected in zip(passage_texts, is_selected):
        all_passages.append(p_text)

        # Save the gold doc index for this query
        if selected == 1 and not found:
            queries.append(query)
            neighbors.append([doc_index])
            found = True

        doc_index += 1

# Save neighbors list (shape: [num_queries, 1])
print(f"Saving gold neighbors for {len(neighbors)} queries")
np.save("embeddings/top1_neighbors.npy", np.array(neighbors, dtype=np.int32))

print("embedding passages")
passage_embeddings = model.encode(all_passages, batch_size=128, show_progress_bar=True)
np.save("embeddings/top1_passage_embeddings.npy", passage_embeddings)

print("embedding queries")
query_embeddings = model.encode(queries, batch_size=128, show_progress_bar=True)
np.save("embeddings/top1_query_embeddings.npy", query_embeddings)
