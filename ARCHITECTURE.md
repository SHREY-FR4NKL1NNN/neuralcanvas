# NeuralCanvas — Architecture

## What is NeuralCanvas

NeuralCanvas is a **real-time neural network training observatory**. You define a
feed-forward architecture, pick a dataset, and train it on the GPU while every
internal detail streams live to the browser: a per-layer network diagram with
gradient-flow pulses, loss/accuracy curves, per-step gradient norms and
dead-neuron percentages, per-epoch weight heatmaps, and activation-distribution
histograms. The problem it solves is **opacity**: a training network is normally a
black box whose internal state you only see through a final loss number.
NeuralCanvas makes that internal state observable as it changes.

## Why this matters

Understanding *why* a network learns what it learns — **mechanistic
interpretability** — is one of the hardest open problems in AI. You cannot
interpret what you cannot observe. NeuralCanvas is a small, honest step toward
observability: it surfaces the quantities interpretability researchers actually
reason about (weights, gradients, activations, dead units) and renders their
*evolution over training* in real time. Watching a ReLU layer's dead-neuron
percentage climb, or a gradient norm vanish toward zero, turns abstract failure
modes into things you can literally see happening.

## The memory layer

NeuralCanvas includes an optional **memory-augmentation** layer, with deliberate
conceptual honesty about what it is and isn't.

- **What it does:** text memories are embedded (sentence-transformers `MiniLM`,
  384-dim, or Ollama `mistral`, 4096-dim) and stored. Before each training batch,
  the string form of a sample's features is embedded into a query; the most
  cosine-similar memories are retrieved; a **similarity-weighted average** of
  their embeddings (the *context vector*) is concatenated onto the network input.
  The first layer is widened to `input_size + memory_dim` to accept it.
- **What real MANNs do:** memory-augmented neural networks — the Neural Turing
  Machine and the Differentiable Neural Computer — use **differentiable** memory
  with learned read/write heads. The addressing mechanism is part of the
  computation graph, so the network *learns how to use memory* end-to-end.
- **What this approximates:** NeuralCanvas uses **non-differentiable cosine
  retrieval**. Gradients do **not** flow back through the retrieval step — the
  memory influences the *input* the network sees, but the network does not learn
  the retrieval policy. This is the one deliberate simplification.
- **Why it's still interesting:** the memory measurably changes *what the network
  learns*. Because the context vector shifts the input distribution, you can watch
  its influence live — in the loss curve, in the activation histograms of the
  first layer, and in the memory-retrieval events flashing in the panel. It makes
  an otherwise-abstract idea (auxiliary context as features) concrete and visible.

## Architecture diagram

```
Text memories  ──► Embeddings (MiniLM | Ollama)  ──►  MemoryStore (cosine)
                                                            │
                                                            │ top-k weighted
                                                            │ retrieval
                                                            ▼
   Input batch ──────────► [ concat ] ──► Network ──► Loss ──► Backward
                              ▲                                   │
                              │                                   │ get_internals()
                         context vector                          ▼
                                                  weights · grad norms · activations
                                                  · dead-neuron %  ──► SSE ──► browser
```

Data flow per session: the **trainer** thread loads the dataset, builds the
network, and runs the loop; after every backward pass it reads `get_internals()`
and the **streamer** fans those out as Server-Sent Events to the frontend, which
distributes them to the visual panels.

## Limitations and honest caveats

- **Non-differentiable memory retrieval.** Gradients do not flow through cosine
  retrieval (see above). This is an approximation of MANNs, not a MANN.
- **Feed-forward only.** No transformers, no attention, no recurrence. The
  visualisations assume a simple linear-layer stack.
- **Gradient-flow visualisation is approximate.** The network diagram pulses by
  gradient *norm* (magnitude), not gradient *direction* — it shows whether
  gradients are healthy/vanishing/exploding, not where they point.
- **Memory backend switch requires a network rebuild.** MiniLM is 384-dim and
  Ollama/mistral is 4096-dim, so switching backends changes `memory_dim` and the
  input layer is rebuilt on the next training start (the UI warns about this).
- **Per-batch embedding cost with memory enabled.** Each batch embeds a query
  string; with the Ollama backend that is one HTTP call per batch, which is slow
  on large datasets. Memory is best demonstrated on XOR / small MNIST.
- **No persistence across restarts.** Sessions, memories, and trained weights live
  in process memory only.

## What I'd build next

- **Differentiable memory** with learned read/write heads (a real NTM/DNC head),
  so gradients flow through addressing and the network learns to use memory.
- **Transformer architecture visualisation** — attention maps, head importance.
- **Gradient *direction* visualisation**, not just norm (e.g. cosine between
  consecutive update directions per layer).
- **Distributed training across multiple GPUs**, visualising per-replica state.
- **Export trained weights and replay training sessions** from a recorded event
  log.
