import subprocess

def run_ann_search(name_dataset, name_embedding, format_file, dataset_docs,dataset_queries, neighbors, algorithm,
                   nclusters, top_k, ells, test_split_percent, split_seed,
                   learner_nunits, learner_nepochs, compute_clusters):
    command = [
        "python", "main_mips.py",
        "--name_dataset", name_dataset,
        "--name_embedding", name_embedding,
        "--format_file", format_file,
        "--dataset_docs", dataset_docs,
        "--dataset_queries", dataset_queries,
        "--dataset_neighbors", neighbors,
        "--algorithm", algorithm,
        "--nclusters", str(nclusters),
        "--top_k", str(top_k),
        "--ells", str(ells),
        "--test_split_percent", str(test_split_percent),
        "--split_seed", str(split_seed),
        "--learner_nunits", str(learner_nunits),
        "--learner_nepochs", str(learner_nepochs),
        "--compute_clusters", str(compute_clusters)

    ]

    print("Running ANN Search with the following command:")
    print(" ".join(command))
    
    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    # Define the parameters
    params = {
        "name_dataset": "Flickr30k",
        "name_embedding": "clip_vit",
        "format_file": "npy",
        "dataset_docs": "embeddings/passages_embeddings.npy",
        "dataset_queries": "embeddings/queries_embeddings.npy",
        "neighbors": "embeddings/top10_neighbors.npy",
        "algorithm": "kmeans-spherical",
        "nclusters": 2966,  # Square root of dataset size
        "top_k": 10,
        "ells": 30,  # Fraction of clusters examined
        "test_split_percent": 20,
        "split_seed": 42,
        "learner_nunits": 0,
        "learner_nepochs": 100,
        "compute_clusters": 0  # Compute clusters (set to 0 to reuse clusters)
    }

    # Run the ANN search
    run_ann_search(**params)
