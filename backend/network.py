"""Configurable feed-forward neural network with full internals introspection.

The whole point of NeuralCanvas is to make a network's internal state observable
while it trains, so :class:`NeuralCanvas` keeps every quantity a visualiser might
want — per-layer weight matrices, gradient norms, pre-activations, and dead-neuron
percentages — and exposes them through :meth:`get_internals`.

Memory augmentation: when ``config.memory_dim > 0`` the first linear layer is
widened to ``input_size + memory_dim`` and the forward pass concatenates the
retrieved memory context vector onto the input,
``torch.cat([x, memory_embedding], dim=1)``, before the first layer. With
``memory_dim == 0`` memory is disabled and the network is a plain MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

_ACTIVATIONS = ("relu", "tanh", "sigmoid", "leaky_relu")


@dataclass
class NetworkConfig:
    """Shape and regularisation of a :class:`NeuralCanvas` network.

    Attributes:
        input_size: Number of raw input features (before memory augmentation).
        hidden_layers: Width of each hidden layer, e.g. ``[64, 32]``.
        output_size: Number of output units (1 for binary/BCE, n_classes for CE).
        activation: One of ``relu``/``tanh``/``sigmoid``/``leaky_relu``.
        dropout: Dropout probability applied after each hidden activation (0–0.5).
        memory_dim: Width of the auxiliary memory vector concatenated to the input
            (``0`` disables memory). The effective first-layer input is
            ``input_size + memory_dim``.
    """

    input_size: int
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    output_size: int = 2
    activation: str = "relu"
    dropout: float = 0.0
    memory_dim: int = 0

    def __post_init__(self) -> None:
        """Validate the configuration, raising ``ValueError`` on bad inputs."""
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}")
        if not 0.0 <= self.dropout <= 0.5:
            raise ValueError("dropout must be in [0.0, 0.5]")
        if self.input_size < 1 or self.output_size < 1:
            raise ValueError("input_size and output_size must be >= 1")
        if self.memory_dim < 0:
            raise ValueError("memory_dim must be >= 0")


class NeuralCanvas(nn.Module):
    """An introspectable MLP whose internals can be read after each forward pass.

    The network is built as an explicit ``ModuleList`` of linear layers so the
    forward pass can record each layer's pre-activation. Hidden layers apply the
    configured activation then dropout; the output layer emits raw logits (the
    loss function handles the final non-linearity).
    """

    def __init__(self, config: NetworkConfig) -> None:
        """Build the linear stack from ``config`` (memory widens the first layer)."""
        super().__init__()
        self.config = config
        effective_input = config.input_size + max(0, config.memory_dim)
        sizes = [effective_input, *config.hidden_layers, config.output_size]
        self.linears = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )
        self.dropout = nn.Dropout(config.dropout)
        # Internal state captured during the forward/backward pass.
        self._pre_activations: list[torch.Tensor] = []

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation function element-wise."""
        if self.config.activation == "relu":
            return torch.relu(x)
        if self.config.activation == "tanh":
            return torch.tanh(x)
        if self.config.activation == "sigmoid":
            return torch.sigmoid(x)
        return torch.nn.functional.leaky_relu(x, negative_slope=0.01)

    def forward(
        self, x: torch.Tensor, memory_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run a forward pass, recording per-layer pre-activations.

        If memory is enabled (``memory_dim > 0``) and ``memory_embedding`` is
        provided, it is concatenated onto ``x`` before the first layer. Returns
        the output-layer logits.
        """
        if self.config.memory_dim > 0 and memory_embedding is not None:
            x = torch.cat([x, memory_embedding], dim=1)

        self._pre_activations = []
        last = len(self.linears) - 1
        for i, linear in enumerate(self.linears):
            x = linear(x)
            # Record the pre-activation (the linear output, before non-linearity).
            self._pre_activations.append(x.detach())
            if i < last:
                x = self._activate(x)
                x = self.dropout(x)
        return x

    def _dead_neuron_pct(self) -> list[float]:
        """Percentage of dead neurons per layer (output 0 across the whole batch).

        Most meaningful for ReLU, where a neuron that never fires for any sample
        in the batch is "dead". Computed from the post-activation of each layer
        (the output layer uses its raw pre-activation, which is rarely exactly 0).
        """
        pct: list[float] = []
        last = len(self.linears) - 1
        for i, pre in enumerate(self._pre_activations):
            post = self._activate(pre) if i < last else pre
            # A neuron is dead if its output is ~0 for every sample in the batch.
            dead = (post.abs() < 1e-6).all(dim=0)
            pct.append(float(dead.float().mean().item() * 100.0))
        return pct

    def get_internals(self) -> dict:
        """Return the network's current internals after a forward/backward pass.

        Keys:
            weight_matrices: list of 2D ``numpy`` arrays, one per linear layer.
            gradient_norms: list of per-layer weight-gradient L2 norms, or ``None``
                if no backward pass has run yet (gradients not populated).
            activations: list of 2D ``numpy`` arrays of pre-activation values
                (batch x units), one per layer, from the most recent forward pass.
            dead_neuron_pct: list of per-layer dead-neuron percentages.
        """
        weight_matrices = [
            linear.weight.detach().cpu().numpy() for linear in self.linears
        ]
        if all(linear.weight.grad is not None for linear in self.linears):
            gradient_norms: list[float] | None = [
                float(linear.weight.grad.detach().norm().item())
                for linear in self.linears
            ]
        else:
            gradient_norms = None
        activations = [pre.cpu().numpy() for pre in self._pre_activations]
        return {
            "weight_matrices": weight_matrices,
            "gradient_norms": gradient_norms,
            "activations": activations,
            "dead_neuron_pct": self._dead_neuron_pct(),
        }

    def layer_labels(self) -> list[float]:
        """Return human labels for each linear layer ("Input"→"Output")."""
        n = len(self.linears)
        labels = []
        for i in range(n):
            if i == n - 1:
                labels.append("Output")
            else:
                labels.append(f"Hidden {i + 1}")
        return labels


def histogram_internals(internals: dict, bins: int = 20, max_dim: int = 32) -> dict:
    """Compress raw internals into a bandwidth-friendly form for the wire.

    Weight matrices are sampled to at most ``max_dim`` x ``max_dim`` (a top-left
    slice) and activations are turned into ``bins``-bin histograms server-side, so
    the per-epoch payload stays small regardless of layer width.
    """
    weights = []
    for w in internals["weight_matrices"]:
        sliced = np.asarray(w)[:max_dim, :max_dim]
        weights.append(sliced.astype(float).tolist())

    activation_hist = []
    for a in internals["activations"]:
        flat = np.asarray(a).reshape(-1)
        if flat.size == 0:
            activation_hist.append({"counts": [], "edges": []})
            continue
        counts, edges = np.histogram(flat, bins=bins)
        activation_hist.append(
            {"counts": counts.astype(int).tolist(), "edges": edges.astype(float).tolist()}
        )

    weight_extremes = []
    for w in internals["weight_matrices"]:
        arr = np.asarray(w)
        weight_extremes.append({"min": float(arr.min()), "max": float(arr.max())})

    return {
        "weight_matrices": weights,
        "weight_extremes": weight_extremes,
        "activation_histograms": activation_hist,
        "gradient_norms": internals["gradient_norms"],
        "dead_neuron_pct": internals["dead_neuron_pct"],
    }
