"""Tests for the configurable network and its internals introspection (CPU)."""

import pytest
import torch

from network import NetworkConfig, NeuralCanvas, histogram_internals


def test_config_validation():
    with pytest.raises(ValueError):
        NetworkConfig(input_size=2, activation="bogus")
    with pytest.raises(ValueError):
        NetworkConfig(input_size=2, dropout=0.9)
    with pytest.raises(ValueError):
        NetworkConfig(input_size=0)


def test_forward_shape_and_internals():
    cfg = NetworkConfig(input_size=2, hidden_layers=[8, 4], output_size=1, activation="relu")
    net = NeuralCanvas(cfg)
    x = torch.randn(16, 2)
    out = net(x)
    assert out.shape == (16, 1)

    internals = net.get_internals()
    # 3 linear layers: 2->8, 8->4, 4->1
    assert len(internals["weight_matrices"]) == 3
    assert len(internals["activations"]) == 3
    assert len(internals["dead_neuron_pct"]) == 3
    assert internals["gradient_norms"] is None  # no backward yet

    out.sum().backward()
    internals = net.get_internals()
    assert internals["gradient_norms"] is not None
    assert len(internals["gradient_norms"]) == 3
    assert all(isinstance(g, float) for g in internals["gradient_norms"])


def test_memory_widens_first_layer():
    cfg = NetworkConfig(input_size=2, hidden_layers=[8], output_size=2, memory_dim=4)
    net = NeuralCanvas(cfg)
    assert net.linears[0].in_features == 6  # 2 + 4
    out = net(torch.randn(5, 2), torch.randn(5, 4))
    assert out.shape == (5, 2)


def test_dead_neuron_pct_range():
    cfg = NetworkConfig(input_size=2, hidden_layers=[8], output_size=2, activation="relu")
    net = NeuralCanvas(cfg)
    net(torch.randn(10, 2))
    pct = net.get_internals()["dead_neuron_pct"]
    assert all(0.0 <= p <= 100.0 for p in pct)


def test_histogram_internals_compression():
    cfg = NetworkConfig(input_size=2, hidden_layers=[40], output_size=2)
    net = NeuralCanvas(cfg)
    net(torch.randn(8, 2)).sum().backward()
    h = histogram_internals(net.get_internals(), bins=20, max_dim=32)
    assert len(h["weight_matrices"]) == 2
    assert all(len(w) <= 32 for w in h["weight_matrices"])  # sampled
    assert len(h["activation_histograms"][0]["counts"]) == 20
    assert "min" in h["weight_extremes"][0] and "max" in h["weight_extremes"][0]
