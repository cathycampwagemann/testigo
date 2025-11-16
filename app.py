from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import requests
import re

# ====== CLAVES ======
OPENAI_API_KEY = "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"  # <--- pon tu clave
DID_API_KEY = "Y2F0eWNhbXBAZ21haWwuY29t:bdpnussf2aNNVCXyQVmyC"
# ====================

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

QA_PAIRS = []
EMB_Q = None


def cosine_sim_vector_to_matrix(query_vec, matrix):
    """
    query_vec: vector (dim,)
    matrix: matriz (n, dim)
    return: array de similitudes (n,)
    """
    q = np.array(query_vec, dtype=float)
    M = np.array(matrix, dtype=float)

    # producto punto entre cada fila de M y q
    num = M @ q
    # norma de cada fila de M
    M_norm = np.linalg.norm(M, axis=1)
    # norma del vector q
    q_norm = np.linalg.norm(q)

    denom = (M_norm * q_norm) + 1e-10  # para evitar división por cero
    return num / denom
    
def embed(texts):
    vec = []
    for t in texts:
        e = client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
        vec.append(e)
    return np.array(vec)

def did_video(text):
    data = {
        "script": {
            "type": "text",
            "provider": {"type": "microsoft", "voice_id": "es-ES-AlvaroNeural"},
            "input": text
        },
        "config": {"fluent": True, "pad_audio": 0.5},
        "source_url": "https://i.ibb.co/Tx4fBvDj/Alessio.jpg"
    }
    h = {"Authorization": f"Basic {DID_API_KEY}"}
    r = requests.post("https://api.d-id.com/talks", headers=h, json=data).json()

    if "id" not in r:
        return {"error": "Error al generar video", "raw": r}

    talk_id = r["id"]

    # Polling
    for _ in range(30):
        status = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=h).json()
        if "result_url" in status and status["result_url"]:
            return status
        import time; time.sleep(1)

    return {"error": "timeout"}


class Load(BaseModel):
    text: str

class Ask(BaseModel):
    question: str


@app.post("/load")
def load(b: Load):
    global QA_PAIRS, EMB_Q
    
    text = b.text.replace("\xa0", " ").replace("\r", " ").strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    QA_PAIRS = []

    for i in range(len(lines)-1):
        if ("Para que" in lines[i]) or ("Respecto" in lines[i]):
            QA_PAIRS.append((lines[i], lines[i+1]))

    questions = [q for q, a in QA_PAIRS]
    EMB_Q = embed(questions)

    return {"ok": True, "pairs": len(QA_PAIRS)}


@app.post("/ask")
def ask(b: Ask):
    global QA_PAIRS, EMB_Q

    qe = client.embeddings.create(model="text-embedding-3-small", input=b.question).data[0].embedding
    sims = cosine_similarity([qe], EMB_Q)[0]
    idx = int(np.argmax(sims))

    if sims[idx] < 0.40:
        return did_video("No dispongo de esa información en mis registros.")

    _, answer = QA_PAIRS[idx]
    return did_video(answer)


FRONT = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Alessio Testigo</title></head>
<body style="font-family:sans-serif;text-align:center;">
<h2>🎤 Testigo AI - Alessio</h2>
<textarea id="base" rows="10" style="width:80%;" placeholder="Pega el texto completo aquí"></textarea><br>
<button onclick="loadBase()">📚 Cargar Base</button><br><br>
<textarea id="q" rows="3" style="width:80%;" placeholder="Haz tu pregunta"></textarea><br>
<button onclick="ask()">Preguntar</button>
<div><video id="v" controls autoplay playsinline style="margin-top:20px;width:60%;"></video></div>

<script>
async function loadBase(){
  let t=document.getElementById('base').value;
  await fetch('/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  document.getElementById('base').style.display='none';
}
async function ask(){
  let q=document.getElementById('q').value;
  let r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  let j=await r.json();
  if(j.result_url) document.getElementById('v').src=j.result_url;
}
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return FRONT

