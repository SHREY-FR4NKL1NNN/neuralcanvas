"""The training loop: runs in a background thread and streams its internals.

A :class:`Trainer` owns one training run. ``start()`` spawns a daemon thread that
loads the dataset, builds the network, and trains — capturing weights, gradient
norms, activations, and dead-neuron percentages after every backward pass and
emitting them as SSE events. Pause/resume/stop are cooperative (checked between
batches). The loss is selected automatically from the output size (BCE for a
single output, cross-entropy otherwise), since the UI configures architecture but
not the loss.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

import datasets as datasets_mod
from embeddings import EmbeddingRouter
from log_config import get_logger
from memory import MemoryStore
from network import NetworkConfig, NeuralCanvas, histogram_internals
from streamer import EventStreamer

logger = get_logger("trainer")

MAX_EPOCHS = 50


@dataclass
class TrainingConfig:
    """All knobs for one training run (mirrors the frontend form + dataset)."""

    network_config: NetworkConfig
    dataset: str = "xor"
    dataset_params: dict = field(default_factory=dict)
    optimizer: str = "adam"
    learning_rate: float = 0.01
    epochs: int = 20
    batch_size: int = 32
    loss_fn: str = "cross_entropy"
    use_memory: bool = False
    memory_backend: str = "sentence-transformers"


@dataclass
class TrainerState:
    """A snapshot of the trainer for the ``/session/state`` endpoint."""

    status: str = "idle"
    epoch: int = 0
    step: int = 0
    steps_per_epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_acc: float = 0.0
    val_acc: float = 0.0
    internals: dict = field(default_factory=dict)
    memory_stats: dict = field(default_factory=dict)


class Trainer:
    """Owns a single training run and its background thread."""

    def __init__(
        self,
        config: TrainingConfig,
        streamer: EventStreamer,
        embeddings: EmbeddingRouter,
        memory: MemoryStore,
    ) -> None:
        """Wire up the trainer with its streamer, embedding router, and memory."""
        self.config = config
        self.streamer = streamer
        self.embeddings = embeddings
        self.memory = memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state = TrainerState()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._resume = threading.Event()
        self._resume.set()  # not paused initially
        self._start_time = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Start training in a daemon background thread (idempotent if running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._resume.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Pause training between batches."""
        self._resume.clear()
        if self.state.status == "training":
            self.state.status = "paused"
            self.streamer.emit("status", {"status": "paused", "message": "Paused"})

    def resume(self) -> None:
        """Resume a paused run."""
        if self.state.status == "paused":
            self.state.status = "training"
            self.streamer.emit("status", {"status": "training", "message": "Resumed"})
        self._resume.set()

    def stop(self) -> None:
        """Request the run to stop; unblocks a paused loop so it can exit."""
        self._stop = True
        self._resume.set()

    def get_state(self) -> TrainerState:
        """Return the current trainer state snapshot."""
        return self.state

    # ---- helpers -----------------------------------------------------------
    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        """Create the configured optimizer over ``params``."""
        lr = self.config.learning_rate
        name = self.config.optimizer.lower()
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        if name == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr)
        return torch.optim.Adam(params, lr=lr)

    def _build_criterion(self, output_size: int) -> tuple[str, nn.Module]:
        """Pick a loss mode + criterion from the output size (and config hint).

        Returns ``(mode, criterion)`` where mode is ``"bce"``/``"cross_entropy"``/
        ``"mse"``. A single output is always BCE-with-logits; otherwise
        cross-entropy, unless MSE is explicitly requested (one-hot targets).
        """
        if output_size == 1:
            return "bce", nn.BCEWithLogitsLoss()
        if self.config.loss_fn == "mse":
            return "mse", nn.MSELoss()
        return "cross_entropy", nn.CrossEntropyLoss()

    def _accuracy(self, mode: str, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Compute batch accuracy for the active loss mode."""
        if mode == "bce":
            preds = (torch.sigmoid(logits) > 0.5).float()
            return float((preds == y).float().mean().item())
        preds = torch.argmax(logits, dim=1)
        targets = y if y.dim() == 1 else torch.argmax(y, dim=1)
        return float((preds == targets).float().mean().item())

    def _memory_embedding(
        self, x: torch.Tensor, input_size: int, memory_dim: int
    ) -> tuple[torch.Tensor, dict | None]:
        """Build the per-batch memory tensor and a retrieval-event payload.

        Embeds the string form of the first sample's features, retrieves a
        similarity-weighted context vector from the store, and broadcasts it to
        the whole batch. Returns ``(memory_tensor, retrieval_info|None)``.
        """
        feats = x[0, :input_size].detach().cpu().numpy()
        query_text = ", ".join(f"{v:.3f}" for v in feats[: min(16, len(feats))])
        query_emb = self.embeddings.embed([query_text])[0]
        context = self.memory.get_context_vector(query_emb)
        if context.shape[0] != memory_dim:  # dimension guard
            context = np.zeros(memory_dim, dtype=np.float32)
        mem_tensor = (
            torch.tensor(context, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(x.shape[0], 1)
        )
        retrieval_info: dict | None = None
        if self.memory.entries:
            top = self.memory.retrieve_with_scores(query_emb, top_k=3)
            retrieval_info = {
                "query_preview": query_text,
                "retrieved_texts": [e.text for e, _ in top],
                "scores": [round(float(s), 3) for _, s in top],
                "context_injected": True,
            }
        return mem_tensor, retrieval_info

    @torch.no_grad()
    def _validate(self, network, val_loader, criterion, mode, use_memory, input_size,
                  memory_dim) -> tuple[float, float]:
        """Run a full validation pass; returns ``(val_loss, val_acc)``."""
        network.eval()
        losses, accs, counts = 0.0, 0.0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            mem = None
            if use_memory:
                mem, _ = self._memory_embedding(xb, input_size, memory_dim)
            logits = network(xb, mem)
            target = yb.float() if mode == "mse" else yb
            if mode == "mse":
                target = torch.nn.functional.one_hot(
                    yb, num_classes=logits.shape[1]
                ).float()
            losses += float(criterion(logits, target).item()) * xb.shape[0]
            accs += self._accuracy(mode, logits, yb) * xb.shape[0]
            counts += xb.shape[0]
        network.train()
        if counts == 0:
            return 0.0, 0.0
        return losses / counts, accs / counts

    # ---- main loop ---------------------------------------------------------
    def _run(self) -> None:
        """The background training loop (entry point of the worker thread)."""
        try:
            self._start_time = time.time()
            self.state.status = "training"
            self.streamer.emit("status", {"status": "training", "message": "Starting"})

            data = datasets_mod.load_dataset(
                self.config.dataset, self.config.dataset_params, self.config.batch_size
            )
            input_size = data["input_size"]
            output_size = data["output_size"]

            # The dataset is the source of truth for I/O shape. Memory widens input.
            net_cfg = self.config.network_config
            net_cfg.input_size = input_size
            net_cfg.output_size = output_size
            use_memory = self.config.use_memory
            if use_memory:
                self.embeddings.switch(self.config.memory_backend)
                net_cfg.memory_dim = self.embeddings.get_dim()
            else:
                net_cfg.memory_dim = 0
            memory_dim = net_cfg.memory_dim

            network = NeuralCanvas(net_cfg).to(self.device)
            network.train()
            optimizer = self._build_optimizer(network.parameters())
            mode, criterion = self._build_criterion(output_size)
            labels = network.layer_labels()

            train_loader = data["train_loader"]
            val_loader = data["val_loader"]
            steps_per_epoch = max(1, len(train_loader))
            emit_every = max(1, steps_per_epoch // 20)
            self.state.steps_per_epoch = steps_per_epoch
            epochs = min(self.config.epochs, MAX_EPOCHS)

            logger.info(
                "Session training: %s, net %s, memory=%s, device=%s",
                data["dataset_name"], net_cfg.hidden_layers, use_memory, self.device,
            )

            global_step = 0
            for epoch in range(1, epochs + 1):
                self.state.epoch = epoch
                for step_in_epoch, (xb, yb) in enumerate(train_loader, start=1):
                    if self._stop:
                        self._finish("stopped")
                        return
                    self._resume.wait()  # blocks while paused
                    if self._stop:
                        self._finish("stopped")
                        return

                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    mem_tensor = None
                    retrieval_info = None
                    if use_memory:
                        mem_tensor, retrieval_info = self._memory_embedding(
                            xb, input_size, memory_dim
                        )

                    optimizer.zero_grad()
                    logits = network(xb, mem_tensor)
                    if mode == "mse":
                        target = torch.nn.functional.one_hot(
                            yb, num_classes=output_size
                        ).float()
                    else:
                        target = yb
                    loss = criterion(logits, target)
                    loss.backward()
                    # Capture internals AFTER backward, BEFORE the optimizer step
                    # (gradients are populated, weights are pre-update).
                    internals = network.get_internals()
                    optimizer.step()

                    global_step += 1
                    self.state.step = step_in_epoch
                    train_loss = float(loss.item())
                    train_acc = self._accuracy(mode, logits.detach(), yb)
                    self.state.train_loss = train_loss
                    self.state.train_acc = train_acc

                    if step_in_epoch % emit_every == 0 or step_in_epoch == steps_per_epoch:
                        self.streamer.emit(
                            "step",
                            {
                                "epoch": epoch,
                                "step": step_in_epoch,
                                "global_step": global_step,
                                "train_loss": train_loss,
                                "val_loss": None,
                                "train_acc": train_acc,
                                "internals_summary": {
                                    "gradient_norms": internals["gradient_norms"],
                                    "dead_neuron_pct": internals["dead_neuron_pct"],
                                    "train_loss": train_loss,
                                    "layer_labels": labels,
                                },
                            },
                        )
                        if retrieval_info is not None:
                            self.streamer.emit("memory_retrieval", retrieval_info)

                # ---- end of epoch ----
                val_loss, val_acc = self._validate(
                    network, val_loader, criterion, mode, use_memory,
                    input_size, memory_dim,
                )
                self.state.val_loss = val_loss
                self.state.val_acc = val_acc
                self.state.memory_stats = self.memory.get_retrieval_stats()
                full = histogram_internals(internals)
                full["layer_labels"] = labels
                self.state.internals = full
                self.streamer.emit(
                    "epoch_end",
                    {
                        "epoch": epoch,
                        "train_loss": self.state.train_loss,
                        "val_loss": val_loss,
                        "train_acc": self.state.train_acc,
                        "val_acc": val_acc,
                        "full_internals": full,
                    },
                )
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                    epoch, epochs, self.state.train_loss, val_loss, val_acc,
                )

            self._finish("done")
        except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
            logger.exception("Training failed")
            self.state.status = "error"
            self.streamer.emit("error", {"message": str(exc)})

    def _finish(self, status: str) -> None:
        """Emit the terminal status/done events and record the final state."""
        self.state.status = status
        total_ms = int((time.time() - self._start_time) * 1000)
        if status == "done":
            self.streamer.emit(
                "done",
                {
                    "final_metrics": {
                        "train_loss": self.state.train_loss,
                        "val_loss": self.state.val_loss,
                        "train_acc": self.state.train_acc,
                        "val_acc": self.state.val_acc,
                    },
                    "total_time_ms": total_ms,
                },
            )
            logger.info("Training done in %d ms", total_ms)
        else:
            self.streamer.emit("status", {"status": status, "message": f"Training {status}"})
            logger.info("Training %s after %d ms", status, total_ms)
