# Face Search

Lean event-photo face search using SCRFD, ArcFace, and FAISS.

## Setup

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

On an NVIDIA machine with CUDA/cuDNN available, install the GPU-enabled ONNX
Runtime wheel in the same environment. It retains CPU fallback:

```bash
./.venv/bin/pip install --upgrade onnxruntime-gpu
```

Put the Hugging Face token in `.env` as `HF_TOKEN=...`. Keep `.env` local.

Download `scrfd_10g.onnx` and `arcface_w600k_r50.onnx` from
`globalnebula/insightface-buffalo-l-onnx` into `models/`, then prepare the
WIDER FACE validation set and deterministic VGGFace2 subset under
`data/validation/` as specified in `codex/plan.md`.

The specified VGGFace2 repository is gated. The Hugging Face account in
`.env` must accept its non-commercial research terms before the subset can be
created.

## Commands

```bash
./.venv/bin/python scripts/tune.py
./.venv/bin/python scripts/build_index.py --image-folder data/event_photos/smoke_event
./.venv/bin/python scripts/search.py --query data/queries/smoke_queries/example.jpg
./.venv/bin/uvicorn app:app --reload
```

The app exposes `/health`, `/events`, `/search`, and a minimal UI at `/`.
Event uploads are persisted and indexed immediately; query selfies are never
added to the FAISS event index.

The checked final parameters are stored in `storage/best_config.json`; the
application and the index/search scripts load that file automatically.
