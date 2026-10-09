# VisionRAG AI — Visual Document Intelligence

Flask full-stack prototype combining **a trainable PyTorch CNN**, OCR/text extraction, semantic retrieval with SentenceTransformers, SQLite storage, and optional **Ollama grounded generation**.

## Features
- Upload PDF, TXT, PNG or JPEG (12 MB maximum).
- Extract PDF text via PyMuPDF; images via Tesseract OCR.
- Train a four-class CNN for invoice/letter/form/report images.
- Chunk text with overlap, embed using all-MiniLM-L6-v2, rank using cosine similarity, return top 3 evidence passages.
- Optional local Ollama response generation with source context.
- Web interface, health endpoint and Docker configuration.

## Run locally
Install Python 3.11+, Tesseract OCR for image uploads, and then:

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. First semantic query downloads a public embedding model.

For grounded generated answers, install/run Ollama and set:
```bash
export OLLAMA_URL=http://localhost:11434
ollama pull llama3.2
python app.py
```

## Train the CNN
Place labeled images in `data/train/{form,invoice,letter,report}/` and `data/val/{form,invoice,letter,report}/`, then run `python train_cnn.py`. The checkpoint is saved to `models/document_cnn.pt`. **Without a trained checkpoint, image classification is explicitly marked unclassified**.

## Architecture
Browser → Flask upload → text extraction / optional CNN → SQLite → text chunking → MiniLM embeddings → cosine retrieval → optional Ollama → cited evidence.

## Engineering limitations / next steps
This is an **MVP, not a validated production system**. There are no pretrained CNN weights or included training dataset. No accuracy claim is made. Add per-user authorization, CSRF protections, rate limits, file malware scanning, PDF OCR fallback, background job queue, persistent vector index, dataset documentation, F1/confusion matrix, retrieval recall@k, CI tests, and cloud object storage before real-world use. Uploaded files may contain sensitive information; do not use confidential documents on a public deployment.

## Interview talking points
Why CNN vs transfer learning? Why overlapping chunks? Why MiniLM? How does cosine ranking work? What is hallucination mitigation? How to measure recall@k and latency? Why asynchronous OCR? How to scale vector search? How to secure multi-tenant document access?
