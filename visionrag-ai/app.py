import os
import sqlite3
import uuid
from pathlib import Path
import fitz
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
DB = BASE / "visionrag.sqlite3"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
embedder = None

def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, filename TEXT, text TEXT, category TEXT)")
    return con

def embed(texts):
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return embedder.encode(texts, normalize_embeddings=True)

def chunks(text, size=850, overlap=150):
    if not text.strip():
        return []
    result = []
    step = size - overlap
    for start in range(0, len(text), step):
        part = text[start:start + size]
        if part.strip():
            result.append(part)
        if start + size >= len(text):
            break
    return result

def extract(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    import pytesseract
    return pytesseract.image_to_string(Image.open(path))

def classify(path):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return "document"
    model_path = BASE / "models" / "document_cnn.pt"
    if not model_path.exists():
        return "unclassified (train CNN first)"
    import torch
    from torchvision import transforms
    from cnn import DocumentCNN, CLASSES
    model = DocumentCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    image = Image.open(path).convert("RGB")
    tensor = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).item()
    return CLASSES[prediction]

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="Missing file"), 400
    name = secure_filename(file.filename)
    extension = Path(name).suffix.lower()
    if extension not in ALLOWED:
        return jsonify(error="Unsupported file type"), 400
    doc_id = uuid.uuid4().hex
    path = UPLOADS / (doc_id + extension)
    file.save(path)
    try:
        content = extract(path)
        category = classify(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        return jsonify(error=f"Could not process document: {exc}"), 422
    if not content.strip():
        return jsonify(error="No readable text detected"), 422
    with db() as con:
        con.execute("INSERT INTO documents VALUES (?,?,?,?)", (doc_id, name, content, category))
    return jsonify(id=doc_id, filename=name, category=category, characters=len(content)), 201

@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    doc_id, question = data.get("document_id"), str(data.get("question", "")).strip()
    if not doc_id or not question:
        return jsonify(error="document_id and question required"), 400
    with db() as con:
        record = con.execute("SELECT text FROM documents WHERE id=?", (doc_id,)).fetchone()
    if record is None:
        return jsonify(error="Document not found"), 404
    passages = chunks(record[0])
    vectors = embed(passages + [question])
    scores = cosine_similarity(vectors[-1:], vectors[:-1])[0]
    selected = np.argsort(scores)[::-1][:3]
    evidence = [{"chunk": int(i), "score": round(float(scores[i]), 4), "text": passages[i]} for i in selected]
    context = "\n\n".join(f"[Source {j+1}] {item['text']}" for j, item in enumerate(evidence))
    answer = None
    if os.getenv("OLLAMA_URL"):
        try:
            response = requests.post(os.environ["OLLAMA_URL"].rstrip("/") + "/api/generate",
                json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": False,
                "prompt": "Answer only from the supplied context. If unknown, say so. Cite source numbers.\nContext:\n" + context + "\nQuestion: " + question},
                timeout=90)
            response.raise_for_status()
            answer = response.json().get("response")
        except requests.RequestException:
            pass
    return jsonify(answer=answer, evidence=evidence, generation_enabled=bool(answer),
        note=None if answer else "No LLM configured or available; returning retrieved evidence.")

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
