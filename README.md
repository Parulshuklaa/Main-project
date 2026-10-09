# InspectIQ — Visual inspection and grounded knowledge assistant

A Flask application combining a **trained, fully NumPy CNN**, hybrid document retrieval, optional LLM generation, session-isolated uploads, and an interactive quality inspection dashboard.

Built by Parul for studying end-to-end AI engineering. **Research demo, not a production defect detector.**

## Run locally

Python 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Choose a clean, scratch, or spot sample, run inspection, then ask the handbook how to handle the predicted pattern. Upload a UTF-8 TXT or text-based PDF to query your own manual. Use no sensitive documents on a public demo.

## What is actually implemented

- Trainable 3×3 convolution (12 filters), ReLU, mean + maximum pooling, linear classification and softmax. Backpropagation updates convolution and classifier weights; no external vision API.
- Reproducible synthetic generator and independently seeded train/validation/test splits. Best validation checkpoint selected before test evaluation.
- Convolution activation visualization, probability distribution, latency and recent inspection history.
- Word/bigram and character TF-IDF retrieval with source IDs, chunking, abstention and source excerpts.
- Optional **generative RAG** using a server-side OpenAI-compatible chat endpoint. Without an API key, the UI explicitly shows retrieval-only mode; excerpts are not represented as LLM answers.
- Flask REST endpoints, SQLite storage with per-session isolation, 24-hour expiry, size/page limits, CSRF tokens, safe DOM rendering and request throttling per session.
- Docker and free Render deployment.

## LLM configuration

Set these server environment variables, never commit keys:

```text
LLM_API_KEY=your-provider-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
SECRET_KEY=long-random-string
COOKIE_SECURE=true
```

Provider generation can incur charges. It is disabled by default. Uploaded retrieved text is sent to the configured provider only when generation is enabled. Source IDs are checked, but citation correctness and factual entailment need further evaluation. Prompt injection mitigation is not a complete defense.

## Evaluate and reproduce

```bash
OPENBLAS_NUM_THREADS=1 python train.py
python -m unittest discover -s tests -v
```

See `artifacts/metrics.json`: 900 training, 300 validation and 300 test samples; 100% accuracy on this easy, same-generator synthetic test. This **does not establish real industrial performance**. Scores are not calibrated. The heatmap is a maximum activation map, **not Grad-CAM**. Test splits vary random seeds, not cameras, materials or factories.

## API

| Endpoint | Purpose |
|---|---|
| GET `/health` | Liveness/model loaded |
| GET `/api/metrics` | Recorded synthetic benchmark |
| POST `/api/inspect` | Multipart `image`: PNG/JPEG |
| POST `/api/documents` | Multipart `document`: TXT/PDF |
| DELETE `/api/documents` | Clear current session uploads |
| POST `/api/ask` | JSON `question` |
| GET `/api/history` | Current session inspections |

POST and DELETE requests need the `X-CSRF-Token` from the homepage. Rate limit is per cookie session and can be bypassed by creating sessions; use gateway limits before production.

## Architecture and scaling

Browser → Flask → NumPy CNN / hybrid retriever → optional LLM → source-backed response. SQLite holds text and prediction metadata; image bytes are not stored. Defaults use `/tmp`, so Render restarts remove history and uploaded documents. One process, four threads; generated secret on restart invalidates old sessions unless configured.

For production: authenticated tenant IDs, Postgres, object storage, asynchronous ingestion queue, separate GPU inference workers, vector database for dense retrieval, reranking, evaluation and tracing, model versioning and staged rollouts. Replace synthetic CNN with a real-data transfer learning or anomaly detection benchmark. Split by item/site/camera to avoid leakage; compare against a simple baseline and measure class-wise precision/recall, AUROC, calibration, latency and OOD failures. Add groundedness and retrieval recall@k tests before enabling unrestricted RAG.

See `docs/INTERVIEW_GUIDE.md` for design decisions and honest interview talking points.
