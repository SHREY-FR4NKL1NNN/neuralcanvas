"""Dataset loaders for NeuralCanvas: XOR, MNIST, and arbitrary CSV uploads.

Each loader returns a uniform dict so the trainer is dataset-agnostic::

    {
        "train_loader": DataLoader, "val_loader": DataLoader,
        "input_size": int, "output_size": int,
        "dataset_name": str, "n_samples": int,
    }

Label convention (so the trainer can pick a loss automatically):
``output_size == 1`` -> binary task, labels are float ``(N, 1)`` for BCE;
``output_size >= 2`` -> multi-class, labels are long ``(N,)`` for cross-entropy.
"""

from __future__ import annotations

import csv
import io
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from log_config import get_logger

logger = get_logger("datasets")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _split_loaders(
    x: np.ndarray,
    y: np.ndarray,
    *,
    binary: bool,
    batch_size: int,
    val_frac: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val ``DataLoader`` pair from feature/label arrays.

    ``binary`` selects the label tensor dtype/shape: float ``(N, 1)`` for BCE
    when True, long ``(N,)`` for cross-entropy when False.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_frac, random_state=42, shuffle=True
    )

    def to_tensors(xa: np.ndarray, ya: np.ndarray) -> TensorDataset:
        xt = torch.tensor(xa, dtype=torch.float32)
        if binary:
            yt = torch.tensor(ya, dtype=torch.float32).reshape(-1, 1)
        else:
            yt = torch.tensor(ya, dtype=torch.long)
        return TensorDataset(xt, yt)

    train = to_tensors(x_train, y_train)
    val = to_tensors(x_val, y_val)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def load_xor(n_samples: int = 1000, noise: float = 0.1, batch_size: int = 32) -> dict:
    """Generate a noisy XOR dataset (2 features, binary label).

    Points are drawn near the four corners of the unit square; the label is the
    XOR of the corner bits, so the classes are not linearly separable. ``noise``
    (0.0–0.3) is the std-dev of Gaussian jitter added to each point.
    """
    noise = float(np.clip(noise, 0.0, 0.3))
    rng = np.random.default_rng(42)
    corners = rng.integers(0, 2, size=(n_samples, 2))
    x = corners.astype(np.float32) + rng.normal(0.0, noise, size=(n_samples, 2)).astype(
        np.float32
    )
    y = (corners[:, 0] ^ corners[:, 1]).astype(np.float32)
    train_loader, val_loader = _split_loaders(
        x, y, binary=True, batch_size=batch_size
    )
    logger.info("Loaded XOR dataset: %d samples, noise=%.2f", n_samples, noise)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "input_size": 2,
        "output_size": 1,
        "dataset_name": "XOR",
        "n_samples": n_samples,
    }


def load_mnist(
    n_samples: int = 5000,
    classes: list[int] | None = None,
    batch_size: int = 32,
) -> dict:
    """Load (a subset of) MNIST, flattened to 784 features normalised to [0, 1].

    Downloads to ``./data`` on first use. ``n_samples`` caps the number of
    training+val examples for speed. ``classes`` restricts to a subset (e.g.
    ``[0, 1]`` for a binary task); labels are then remapped to ``0..k-1`` and the
    output size becomes ``len(classes)``.
    """
    from torchvision import datasets as tv_datasets  # local import: heavy

    os.makedirs(DATA_DIR, exist_ok=True)
    mnist = tv_datasets.MNIST(root=DATA_DIR, train=True, download=True)
    x_all = mnist.data.reshape(len(mnist), -1).numpy().astype(np.float32) / 255.0
    y_all = mnist.targets.numpy().astype(np.int64)

    if classes:
        mask = np.isin(y_all, classes)
        x_all, y_all = x_all[mask], y_all[mask]
        remap = {c: i for i, c in enumerate(sorted(classes))}
        y_all = np.array([remap[int(c)] for c in y_all], dtype=np.int64)
        output_size = len(classes)
    else:
        output_size = 10

    if n_samples and n_samples < len(x_all):
        idx = np.random.default_rng(42).choice(len(x_all), n_samples, replace=False)
        x_all, y_all = x_all[idx], y_all[idx]

    train_loader, val_loader = _split_loaders(
        x_all, y_all, binary=False, batch_size=batch_size
    )
    name = f"MNIST ({','.join(map(str, classes))})" if classes else "MNIST"
    logger.info("Loaded %s: %d samples, %d classes", name, len(x_all), output_size)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "input_size": 784,
        "output_size": output_size,
        "dataset_name": name,
        "n_samples": int(len(x_all)),
    }


def load_csv(csv_bytes: bytes, batch_size: int = 32) -> dict:
    """Load a CSV where the last column is the label and the rest are features.

    Features are min-max normalised to [0, 1] per column; labels are detected
    automatically and remapped to ``0..k-1``. A numeric header row, if present,
    is skipped. Returns ``output_size = n_classes``.
    """
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8")))
    rows = [r for r in reader if r]
    if not rows:
        raise ValueError("CSV is empty")

    # Skip a header row if the first row isn't fully numeric.
    try:
        [float(v) for v in rows[0]]
    except ValueError:
        rows = rows[1:]
    if not rows:
        raise ValueError("CSV has no data rows")

    data = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    x = data[:, :-1].astype(np.float32)
    raw_labels = data[:, -1]

    # Min-max normalise each feature column to [0, 1] (guard zero-range columns).
    col_min = x.min(axis=0)
    col_max = x.max(axis=0)
    span = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    x = ((x - col_min) / span).astype(np.float32)

    classes = sorted({float(v) for v in raw_labels})
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[float(v)] for v in raw_labels], dtype=np.int64)
    output_size = len(classes)
    if output_size < 2:
        raise ValueError("CSV must contain at least 2 label classes")

    train_loader, val_loader = _split_loaders(
        x, y, binary=False, batch_size=batch_size
    )
    logger.info(
        "Loaded CSV: %d samples, %d features, %d classes",
        len(x),
        x.shape[1],
        output_size,
    )
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "input_size": int(x.shape[1]),
        "output_size": output_size,
        "dataset_name": "CSV upload",
        "n_samples": int(len(x)),
    }


def load_dataset(name: str, params: dict, batch_size: int) -> dict:
    """Dispatch to the right loader by ``name`` ("xor"|"mnist"|"csv")."""
    name = name.lower()
    if name == "xor":
        return load_xor(
            n_samples=int(params.get("n_samples", 1000)),
            noise=float(params.get("noise", 0.1)),
            batch_size=batch_size,
        )
    if name == "mnist":
        return load_mnist(
            n_samples=int(params.get("n_samples", 5000)),
            classes=params.get("classes"),
            batch_size=batch_size,
        )
    if name == "csv":
        csv_b64 = params.get("csv_b64")
        if not csv_b64:
            raise ValueError("CSV dataset requires 'csv_b64' in dataset_params")
        import base64

        return load_csv(base64.b64decode(csv_b64), batch_size=batch_size)
    raise ValueError(f"Unknown dataset: {name}")
