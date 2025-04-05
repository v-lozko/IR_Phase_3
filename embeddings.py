import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np

model = SentenceTransformer("sentence-transformers/all_MiniLM-L6-v2")
msmarco_train = load_dataset("ms_marco", "v2.1", split="train")
os.makedirs("embeddings", exist_ok=True)

queries = []
all_passages = []

for example in msmarco_train:
    queries.append(example["query"])
    for p in example["passages"]:
        all_passages.append(p["passage_text"])

print("embedding passages")
passage_embeddings = model.encode(all_passages, batch_size=128, show_progress_bar=True)
np.save("embeddings/passages_embeddings.npy", passage_embeddings)

print("embedding queries")
query_embeddings = model.encode(queries, batch_size=128, show_progress_bar=True)
np.save("embeddings/queries_embeddings.npy", query_embeddings)

print("making nearest neighbors")
scores = query_embeddings @ passage_embeddings.T
top_k = 10
neighbors = np.argsort(-scores, axis =1)[:, :top_k]
np.save("embeddings/top10_neighbors.npy", neighbors)
