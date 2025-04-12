"""

"""

import numpy as np
import pandas as pd
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

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
    # w/out hidden layer
    if n_units == 0:
        model = models.Sequential()
        model.add(layers.Dense(k, activation='softmax', use_bias=False, input_shape=input_shape))
        return model

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
                          n_clusters, n_epochs, lr=1e-3, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input numpy arrays to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    centroids = torch.tensor(centroids, dtype=torch.float32).to(device)

    # Create regression targets from centroid labels
    target_train = centroids[y_train]  # shape: [N, D]
    target_val = centroids[y_val]

    # Create dataloaders
    train_loader = DataLoader(TensorDataset(x_train, target_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, target_val), batch_size=batch_size)

    # Linear model without bias
    model = nn.Linear(x_train.shape[1], centroids.shape[1], bias=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_model_state = None
    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1}: val_loss={val_loss:.5f}")

    # Load best model state
    model.load_state_dict(best_model_state)

    # Use the learned projection to produce new centroids (cluster representatives)
    projected_centroids = model(torch.eye(n_clusters, device=device)).detach().cpu().numpy()
    return projected_centroids
