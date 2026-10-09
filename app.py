import io,json,os,secrets,sqlite3,time
from pathlib import Path
from flask import Flask,request,jsonify,render_template,session,send_file
from PIL import Image,UnidentifiedImageError
import numpy as np
from inspectiq.vision import CNN,sample,LABELS
from inspectiq.retrieval import answer,chunk
ROOT=Path(__file__).parent
app=Flask(__name__); app.secret_key=os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config.update(MAX_CONTENT_LENGTH=5*1024*1024,SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Lax',SESSION_COOKIE_SECURE=os.getenv('COOKIE_SECURE','false')=='true')
DB=os.getenv('DATABASE_PATH','/tmp/inspectiq.db'); model=CNN.load()
Image.MAX_IMAGE_PIXELS=12_000_000

def connection():
    c=sqlite3.connect(DB,timeout=10); c.row_factory=sqlite3.Row
    c.execute('CREATE TABLE IF NOT EXISTS documents (owner TEXT, name TEXT, text TEXT, created REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS events (owner TEXT, label TEXT, confidence REAL, created REAL)')
    c.execute('CREATE TABLE IF NOT EXISTS limits (owner TEXT, bucket INTEGER, count INTEGER, PRIMARY KEY(owner,bucket))')
    c.execute('DELETE FROM documents WHERE created < ?', (time.time()-86400,)); c.execute('DELETE FROM events WHERE created < ?', (time.time()-86400,)); c.execute('DELETE FROM limits WHERE bucket < ?', (int(time.time()//60)-2,)); c.commit(); return c
@app.before_request
def identify():
    if 'id' not in session: session['id']=secrets.token_hex(16)
    if request.path.startswith('/api/') and request.method=='POST':
        if request.headers.get('X-CSRF-Token')!=session.get('csrf'): return jsonify(error='Refresh the page before submitting.'),403
        with connection() as c:
            bucket=int(time.time()//60)
            c.execute('INSERT INTO limits VALUES (?,?,1) ON CONFLICT(owner,bucket) DO UPDATE SET count=count+1',(session['id'],bucket)); c.commit()
            if c.execute('SELECT count FROM limits WHERE owner=? AND bucket=?',(session['id'],bucket)).fetchone()[0]>20: return jsonify(error='Too many requests. Try again in a minute.'),429
@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options']='nosniff'; response.headers['X-Frame-Options']='DENY'; response.headers['Referrer-Policy']='same-origin'; return response
@app.get('/')
def index():
    session.setdefault('csrf',secrets.token_hex(24)); return render_template('index.html',csrf=session['csrf'])
@app.get('/health')
def health(): return jsonify(status='ok',model='numpy-cnn-v1')
@app.get('/api/metrics')
def metrics(): return jsonify(json.loads((ROOT/'artifacts/metrics.json').read_text()))
@app.get('/api/history')
def history():
    with connection() as c: return jsonify([dict(v) for v in c.execute('SELECT label,confidence,created FROM events WHERE owner=? ORDER BY created DESC LIMIT 20',(session['id'],))])
@app.get('/sample/<label>')
def example(label):
    if label not in LABELS: return jsonify(error='Unknown sample'),404
    a=sample(LABELS.index(label),np.random.default_rng(105)); b=io.BytesIO(); Image.fromarray((a*255).astype('uint8')).resize((256,256)).save(b,format='PNG'); b.seek(0)
    return send_file(b,mimetype='image/png')
@app.post('/api/inspect')
def inspect():
    started=time.perf_counter(); f=request.files.get('image')
    if not f: return jsonify(error='Choose a PNG or JPEG image.'),400
    try:
        im=Image.open(f.stream)
        if im.format not in ['PNG','JPEG']: return jsonify(error='PNG and JPEG only.'),400
        im.load(); result=model.predict(im)
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning): return jsonify(error='Invalid or oversized image.'),400
    result['latency_ms']=round((time.perf_counter()-started)*1000,1); result['decision']='Human review required'
    with connection() as c: c.execute('INSERT INTO events VALUES (?,?,?,?)',(session['id'],result['label'],result['confidence'],time.time())); c.commit()
    return jsonify(result)
@app.post('/api/documents')
def documents():
    f=request.files.get('document')
    if not f: return jsonify(error='Choose a TXT or PDF document.'),400
    name=Path(f.filename or '').name[:100]
    try:
        if name.lower().endswith('.pdf'):
            from pypdf import PdfReader
            reader=PdfReader(f.stream)
            if len(reader.pages)>30: return jsonify(error='Maximum 30 pages.'),400
            text='\n'.join(p.extract_text() or '' for p in reader.pages)
        elif name.lower().endswith('.txt'): text=f.read().decode('utf-8')
        else: return jsonify(error='TXT and text-based PDF only.'),400
    except Exception: return jsonify(error='Could not read document. Use a text-based PDF or UTF-8 TXT.'),400
    if len(text)>60000: return jsonify(error='Maximum 60,000 characters.'),400
    if len(text.strip())<20: return jsonify(error='No readable text. Scanned PDFs need OCR.'),400
    chunks=chunk(text)
    with connection() as c:
        count=c.execute('SELECT count(*) FROM documents WHERE owner=?',(session['id'],)).fetchone()[0]
        if count+len(chunks)>120: return jsonify(error='Session document limit reached. Clear your documents.'),400
        c.executemany('INSERT INTO documents VALUES (?,?,?,?)',[(session['id'],name,t,time.time()) for t in chunks]); c.commit()
    return jsonify(name=name,chunks=len(chunks))
@app.delete('/api/documents')
def clear():
    if request.headers.get('X-CSRF-Token')!=session.get('csrf'): return jsonify(error='Invalid token'),403
    with connection() as c: c.execute('DELETE FROM documents WHERE owner=?',(session['id'],)); c.commit()
    return jsonify(cleared=True)
@app.post('/api/ask')
def ask():
    data=request.get_json(silent=True) or {}; q=data.get('question','')
    if not isinstance(q,str) or not 3<=len(q)<=1000: return jsonify(error='Question must be 3–1000 characters.'),400
    with connection() as c: docs=[(r['name'],r['text']) for r in c.execute('SELECT name,text FROM documents WHERE owner=?',(session['id'],))]
    return jsonify(answer(q,docs))
@app.errorhandler(413)
def oversized(e): return jsonify(error='Upload limit is 5 MB.'),413
if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT',5000)))
