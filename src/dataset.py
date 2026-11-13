"""Dataset sintético de clasificación y partición entrenamiento/validación."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_classification.csv"
N_FEATURES = 4
N_CLASSES = 3


def generate_classification_data(
    n_samples: int = 900,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera nubes gaussianas separables, una por clase."""
    rng = np.random.default_rng(seed)
    means = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.4, 1.6, -1.2, 0.8],
            [-1.8, 2.2, 1.5, -1.4],
        ],
        dtype=np.float64,
    )
    if n_classes != means.shape[0] or n_features != means.shape[1]:
        raise ValueError("Esta generación fija usa 4 features y 3 clases.")

    per_class = n_samples // n_classes
    features = []
    labels = []
    for label in range(n_classes):
        samples = means[label] + rng.normal(scale=0.85, size=(per_class, n_features))
        features.append(samples)
        labels.append(np.full(per_class, label, dtype=np.int64))

    x = np.vstack(features)
    y = np.concatenate(labels)
    order = rng.permutation(len(y))
    return x[order], y[order]


def save_dataset(path: Path, features: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"f{i}" for i in range(features.shape[1])] + ["label"])
    table = np.column_stack([features, labels.astype(np.float64)])
    np.savetxt(path, table, delimiter=",", header=header, comments="")


def load_raw_dataset(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        features, labels = generate_classification_data()
        save_dataset(path, features, labels)
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    return table[:, :-1], table[:, -1].astype(np.int64)


def stratified_split(
    labels: np.ndarray, val_ratio: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in np.unique(labels):
        index = np.where(labels == label)[0]
        rng.shuffle(index)
        n_val = max(1, int(round(len(index) * val_ratio)))
        val_parts.append(index[:n_val])
        train_parts.append(index[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def standardize(
    train_features: np.ndarray, val_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza con media y desvío del conjunto de entrenamiento."""
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (train_features - mean) / std, (val_features - mean) / std


def make_loaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    path: Path = DATA_PATH,
) -> tuple[DataLoader, DataLoader, int, int]:
    features, labels = load_raw_dataset(path)
    train_index, val_index = stratified_split(labels, val_ratio=val_ratio, seed=seed)
    x_train, x_val = standardize(features[train_index], features[val_index])
    y_train, y_val = labels[train_index], labels[val_index]

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    input_dim = int(features.shape[1])
    num_classes = int(len(np.unique(labels)))
    return train_loader, val_loader, input_dim, num_classes
