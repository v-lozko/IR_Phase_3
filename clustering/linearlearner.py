"""

"""

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from absl import flags
import torch.nn.functional as F
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def nn_linear(k, input_shape, n_units):
    """
    Neural network model definition, with one input layer (dimension of the embedding),
    one hidden layer (optional) and one output layer (number of clusters).

    :param k: The number of clusters.
    :param input_shape: The dimension of the embedding.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The neural network model.
    """

    if n_units == 0:
        model = models.Sequential()
        model.add(layers.Dense(k, activation='softmax', use_bias=False, input_shape=input_shape))

    # w/ hidden layer
    else:
        model = models.Sequential()
        model.add(layers.Dense(n_units, activation=None, use_bias=False, input_shape=input_shape))
        model.add(layers.Dense(k, activation='softmax', use_bias=False))

    return model

def run_nn(n_clusters, input_shape, x_train, y_train, x_val, y_val, n_epochs=50, n_units=0):
    """
    Trains the neural network.

    :param n_clusters: The number of clusters.
    :param input_shape: The dimension of the embedding.
    :param x_train, y_train: Train set.
    :param x_val, y_val: Validation set.
    :param n_epochs: The number of epochs to train our neural network.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The neural network model and the history of the neural network model.
    """
    # model definition
    model_nn = nn_linear(n_clusters, input_shape, n_units)

    # compiling the model
    model_nn.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

    # training the model
    history = model_nn.fit(x_train, y_train,
                           epochs=n_epochs,
                           batch_size=512,
                           validation_data=(x_val, y_val))

    return model_nn, history


def run_linear_learner(x_train, y_train, x_val, y_val, train_queries, n_clusters, n_epochs, n_units):
    """
    Main function to run the linear-learner algorithm (our method).

    :param x_train, y_train: Train set.
    :param x_val, y_val: Validation set.
    :param train_queries: The queries used to train our model.
    :param n_clusters: The number of clusters.
    :param n_epochs: The number of epochs to train our neural network.
    :param n_units: The number of units in the hidden layer;
     if it is set to 0, the hidden layer is removed, having only the input and output layer.
    :return: The new computed centroid, for each cluster.
    """
    # neural network model
    print('Running linear learner, with number of units: ', n_units)
    _, history = run_nn(n_clusters,
                        (train_queries.shape[1],),
                        x_train, y_train,
                        x_val, y_val,
                        n_epochs=n_epochs,
                        n_units=n_units)

    df_history = pd.DataFrame(history.history)
    with open('history_nn.json', mode='w') as f:
        df_history.to_json(f)

    # best number of epochs
    best_n_epochs = list(range(n_epochs))[np.argmin(history.history['val_loss'])]
    print('Best number of epochs: ', best_n_epochs)

    model_nn, _ = run_nn(n_clusters,
                         (train_queries.shape[1],),
                         x_train, y_train,
                         x_val, y_val,
                         n_epochs=best_n_epochs,
                         n_units=n_units)

    # return the new centroids
    if n_units == 0:
        return model_nn.get_weights()[0].T
    else:
        return np.matmul(model_nn.get_weights()[0], model_nn.get_weights()[1]).T


def run_euclidean_learner(x_train, y_train, x_val, y_val, centroids,
                          n_epochs, lr=1e-3, batch_size=256, hidden_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)
    centroids = torch.tensor(centroids, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    # Learner model: project queries into same space as centroids
    class EuclideanProjectionMLP(nn.Module):
        def __init__(self, input_dim=384, hidden_dim=256, output_dim=384):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = F.relu(self.hidden(x))
            return self.output(x)

    model = EuclideanProjectionMLP(x_train.shape[1], hidden_dim, centroids.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model_state = None
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()

            query_proj = F.normalize(model(xb), p=2, dim=1)  # [B, D]
            centroids_norm = F.normalize(centroids, p=2, dim=1)  # [K, D]

            # Compute distances to all centroids
            q_norm = query_proj.pow(2).sum(dim=1, keepdim=True)  # [B, 1]
            c_norm = centroids_norm.pow(2).sum(dim=1).unsqueeze(0)    # [1, K]
            dot = query_proj @ centroids_norm.T                       # [B, K]
            dists = q_norm + c_norm - 2 * dot                    # [B, K]

            logits = -dists  # negative distances → similarity
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        ranks = []
        with torch.no_grad():
            for xb, yb in val_loader:
                query_proj = F.normalize(model(xb), p=2, dim=1)  # [B, D]
                centroids_norm = F.normalize(centroids, p=2, dim=1)  # [K, D]

                q_norm = query_proj.pow(2).sum(dim=1, keepdim=True)
                c_norm = centroids_norm.pow(2).sum(dim=1).unsqueeze(0)
                dot = query_proj @ centroids_norm.T
                dists = q_norm + c_norm - 2 * dot

                for i in range(xb.size(0)):
                    rank = torch.argsort(dists[i]).tolist().index(yb[i].item())
                    ranks.append(rank)

                logits = -dists
                val_loss += F.cross_entropy(logits, yb).item()

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss /= len(val_loader)
        acc = correct / total
        print(f"Epoch {epoch+1}: val_loss={val_loss:.5f}, val_acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)

    plt.figure(figsize=(8, 5))
    plt.hist(ranks, bins=50, range=(0, 500))
    plt.title("Histogram of True Cluster Ranks")
    plt.xlabel("True Cluster Rank")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cluster_rank_histogram.png")
    print("[INFO] Saved cluster rank histogram to cluster_rank_histogram.png")

    ranks_np = np.array(ranks)
    print(f"[STATS] Top-1 accuracy: {(ranks_np <= 0).mean():.4f}")
    print(f"[STATS] Top-5 accuracy: {(ranks_np < 5).mean():.4f}")
    print(f"[STATS] Top-10 accuracy: {(ranks_np < 10).mean():.4f}")
    print(f"[STATS] Top-30 accuracy: {(ranks_np < 30).mean():.4f}")
    print(f"[STATS] Top-100 accuracy: {(ranks_np < 100).mean():.4f}")

    return model


