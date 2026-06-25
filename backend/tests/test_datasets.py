"""Tests for the XOR and CSV dataset loaders (MNIST is exercised in the live UI)."""

from datasets import load_csv, load_dataset, load_xor


def test_xor_shapes_and_binary_labels():
    d = load_xor(n_samples=200, noise=0.1, batch_size=16)
    assert d["input_size"] == 2
    assert d["output_size"] == 1
    xb, yb = next(iter(d["train_loader"]))
    assert xb.shape[1] == 2
    assert yb.shape[1] == 1  # binary -> float (N, 1)


def test_csv_detects_features_and_classes():
    csv = (
        b"f1,f2,label\n0,0,0\n1,1,1\n0,1,0\n1,0,1\n"
        b"0.2,0.1,0\n0.9,0.8,1\n0.1,0.9,0\n0.8,0.2,1\n"
    )
    d = load_csv(csv, batch_size=4)
    assert d["input_size"] == 2
    assert d["output_size"] == 2
    assert d["n_samples"] == 8


def test_dispatch_xor():
    d = load_dataset("xor", {"n_samples": 100, "noise": 0.05}, 16)
    assert d["dataset_name"] == "XOR"
    assert d["input_size"] == 2
