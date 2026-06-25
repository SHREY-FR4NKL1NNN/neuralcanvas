# NeuralCanvas

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](#)

**A real-time neural network training observatory with a memory layer.** Define
an architecture, train it on the GPU, and watch every internal detail stream live
to the browser — weight heatmaps, gradient norms, activation histograms, loss
curves, and an animated network diagram. Add text memories and watch their
retrieved context shift what the network learns.

🔗 **Live demo:** _(deploy URL placeholder)_ · 📐 **[ARCHITECTURE.md](ARCHITECTURE.md)**

## Highlights

- **Live internals over SSE** — gradient norms and dead-neuron % every step;
  weights, activations, and histograms every epoch.
- **Animated network diagram** — edge thickness = weight magnitude, gradient-flow
  pulses coloured by health (blue healthy / red exploding / grey vanishing).
- **Memory augmentation** — embed text (sentence-transformers or Ollama), retrieve
  by cosine similarity, inject the context vector into the input layer. (An honest
  *approximation* of MANNs — see [ARCHITECTURE.md](ARCHITECTURE.md).)
- **Three datasets** — XOR, MNIST (full or binary subset), and CSV upload.

## Quick start

Backend (Python 3.12, GPU via PyTorch/CUDA):

```bash
cd neuralcanvas/backend
pip install -r requirements.txt --break-system-packages
uvicorn main:app --reload --port 8001      # 8001 avoids LocalMind's 8000
```

Frontend (Node 18+):

```bash
cd neuralcanvas/frontend
npm install
npm run dev                                # serves on http://localhost:5174
```

Open **http://localhost:5174**. The frontend talks to the backend at
`http://localhost:8001` by default (override with `VITE_API_URL`).

## How to use

1. **Configure** — pick a dataset (start with XOR, 1000 samples, noise 0.1), build
   a hidden-layer stack (e.g. `2 → 16 → 8 → 1`), choose activation/optimizer/lr.
2. **Train** — hit **Start**. The loss curve, gradient bars, and network diagram
   update in real time; the diagram pulses on every step.
3. **Observe** — switch the weight-heatmap and activation-histogram layer
   selectors per epoch; watch dead-neuron % and gradient health in the diagram.
4. **Add memories** — enable *Memory augmentation*, type a few text memories
   (e.g. "XOR is a binary classification problem"), and retrain. Retrieval events
   flash in the Memory panel; the influence shows up in the loss/activation views.
5. **Compare** — toggle the embedding backend (sentence-transformers ↔ Ollama),
   change the architecture, and watch how training behaviour changes.

## Endpoints

`POST /session/{start,pause,resume,stop}` · `GET /session/state` ·
`GET /events/stream` (SSE) · `GET /events/history` ·
`POST /memory/{add,clear}` · `GET /memory/list` · `POST /embedding/switch` ·
`GET /health`. Interactive docs at `/docs`.

## Project layout

```
neuralcanvas/
├── backend/
│   ├── main.py        # FastAPI app: sessions, SSE, memory, health
│   ├── trainer.py     # background training loop + internals capture
│   ├── network.py     # configurable MLP + get_internals()
│   ├── datasets.py    # XOR / MNIST / CSV loaders
│   ├── memory.py      # cosine-retrieval memory store (MANN approximation)
│   ├── embeddings.py  # sentence-transformers + Ollama backends
│   ├── streamer.py    # thread-safe SSE fan-out + ring buffer
│   └── log_config.py
├── frontend/          # Vite + React (recharts), 8 visual panels
├── ARCHITECTURE.md
└── README.md
```

## Out of scope (by design)

No transformers/attention, no recurrence, no *differentiable* memory, no user
accounts, no cross-restart persistence, no Docker. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the honest caveats and what's next.
