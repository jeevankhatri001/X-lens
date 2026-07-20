# X-Lens

Research-oriented visual assistance backend with image-quality gating, optional Qwen2.5-VL inference, and optional FAISS RAG.

## What already works
- FastAPI lifecycle and readiness health endpoint
- Request ID middleware
- `/api/v1/analyze`, `/chat`, `/health`, `/metrics`, `/knowledge/status`
- Multi-method blur score: Laplacian, Tenengrad and FFT
- Brightness, contrast, noise, resolution, exposure and colorfulness metrics
- Minimum, geometric-mean and weighted-average aggregation
- Early recapture decision before costly VLM inference
- Qwen adapter and RAG modules as optional dependencies
- Unit/integration tests and an ablation script

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

## Test
```bash
pytest -q
```

## Enable the real VLM
An NVIDIA GPU with adequate VRAM is strongly recommended.
```bash
pip install -e ".[vlm]"
# set XLENS_ENABLE_VLM=true in .env
```

## Enable RAG
```bash
pip install -e ".[rag]"
# set XLENS_ENABLE_RAG=true after building an index
```

## Example API call
```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyze -F "file=@images/test.jpg"
```

## Next implementation milestones
1. Add college URL configuration and crawl allow-list.
2. Build index-management endpoints with authentication.
3. Add optional reranker and Playwright fallback.
4. Calibrate quality thresholds with labelled ESP32-CAM images.
5. Add ESP32 firmware client and audio/text-to-speech output.

The default quality thresholds are starting points, not validated scientific cut-offs. Calibrate them on your own labelled dataset before reporting research results.
