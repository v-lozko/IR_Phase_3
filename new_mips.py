import numpy as np

from embeddings import neighbors

labels = np.load("Flickr30k_clip_vit_kmeans-spherical_label_clustering.npy")
neighbors = np.load("embeddings/neighbors.npy")
print("label_clustering shape:", labels.shape)
print("neighbors shape:", neighbors.shape)
print("Example i_doc:", neighbors[0])
print("label_clustering[i_doc]:", labels[neighbors[0]])