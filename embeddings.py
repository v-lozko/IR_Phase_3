import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = model.to(device)
msmarco_train = load_dataset("ms_marco", "v2.1", split="train")
os.makedirs("embeddings", exist_ok=True)

def compute_topk_neighbors(query_embeds, doc_embeds, top_k=10, batch_size=256):
    num_queries = query_embeds.shape[0]
    neighbors = np.zeros((num_queries, top_k), dtype=np.int32)

    for i in tqdm(range(0, num_queries, batch_size), desc="Computing top-k neighbors"):
        # Get a batch of queries
        q_batch = query_embeds[i:i + batch_size]  # (batch_size, dim)

        # Compute dot products with all document embeddings
        scores = q_batch @ doc_embeds.T  # (batch_size, num_docs)

        # Get top-k indices for each query
        topk = np.argsort(-scores, axis=1)[:, :top_k]  # (batch_size, top_k)

        # Save to the correct slice in the result
        neighbors[i:i + batch_size] = topk

    return neighbors


queries = []
all_passages = []

for example in msmarco_train:
    queries.append(example["query"])

    for i, selected in enumerate(example["passages"]["is_selected"]):
        if selected == 1:
            text = example["passages"]["passage_text"][i]
            all_passages.append(text)

print("embedding passages")
passage_embeddings = model.encode(all_passages, batch_size=128, show_progress_bar=True)
np.save("embeddings/passages_embeddings.npy", passage_embeddings)

print("embedding queries")
query_embeddings = model.encode(queries, batch_size=128, show_progress_bar=True)
np.save("embeddings/queries_embeddings.npy", query_embeddings)

print("making nearest neighbors")
neighbors = compute_topk_neighbors(query_embeddings,passage_embeddings)
np.save("embeddings/top10_neighbors.npy", neighbors)
