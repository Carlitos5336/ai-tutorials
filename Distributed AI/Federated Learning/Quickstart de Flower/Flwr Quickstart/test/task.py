"""Test: A Flower / sklearn app."""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

fds = None  # Cache FederatedDataset

def load_data(partition_id: int, num_partitions: int, dataset_path="data.csv"):
    """
    Load a custom dataset from a local CSV file and partition it.
    
    Args:
        partition_id (int): The ID of the partition assigned to this client.
        num_partitions (int): Total number of clients (partitions).
        dataset_path (str): Path to the local dataset file.
    
    Returns:
        X_train, X_test, y_train, y_test: Partitioned dataset for training and testing.
    """
    
    # Load dataset
    df = pd.read_csv(dataset_path)  # Load data from a CSV file

    # Assume the last column is the target (y), and the rest are features (X)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Labels

    # Shuffle and split dataset into `num_partitions` parts
    X_splits = np.array_split(X, num_partitions)
    y_splits = np.array_split(y, num_partitions)

    # Get the data partition assigned to this client
    X_partition, y_partition = X_splits[partition_id], y_splits[partition_id]

    # Further split the partition into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X_partition, y_partition, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def get_model(penalty: str, local_epochs: int):

    return LogisticRegression(
        penalty=penalty,
        max_iter=local_epochs,
        warm_start=True,
    )


def get_model_params(model):
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [model.coef_]
    return params


def set_model_params(model, params):
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model):
    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))
