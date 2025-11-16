from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import requests
import re
import os
import time

# ====== CLAVES (en producción mejor usar variables de entorno) ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "TU_CLAVE_AQUI_SI_ESTÁS_EN_LOCAL")
DID_API_KEY = os.getenv("DID_API_KEY", "TU_CLAVE_DID_AQUI_SI_ESTÁS_EN_LOCAL")
# ===================================================================

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

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

    if M.size == 0:
        return np.array([])

    num = M @ q                     # producto punto
    M_norm = np.linalg.norm(M, axis=1)
    q_norm = np.linalg.norm(q)

    denom = (M_norm * q_norm) + 1e-10  # para evitar división por cero
    return num / denom


def embed(texts):
    vec = []
    for t in texts:
        e = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        ).data[0].embedding
        vec.append(e)
    return np.array(vec)


def did_video(text):
    data = {
        "script": {
            "type": "text",
            "provider": {"type": "microsoft", "voice_id": "es-US-AlonsoNeural"},
            "input": text.strip()
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
        if "result_url" in status and status.get("result_url"):
            return status
        time.sleep(1)

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

    for i in range(len(lines) - 1):
        if ("Para que" in lines[i]) or ("Respecto" in lines[i]):
            QA_PAIRS.append((lines[i], lines[i + 1]))

    questions = [q for q, a in QA_PAIRS]
    if questions:
        EMB_Q = embed(questions)
    else:
        EMB_Q = np.array([])

    return {"ok": True, "pairs": len(QA_PAIRS)}


@app.post("/ask")
def ask(b: Ask):
    global QA_PAIRS, EMB_Q

    if EMB_Q is None or EMB_Q.size == 0 or not QA_PAIRS:
        return did_video("No dispongo de esa información en mis registros.")

    qe = client.embeddings.create(
        model="text-embedding-3-small",
        input=b.question
    ).data[0].embedding

    sims = cosine_sim_vector_to_matrix(qe, EMB_Q)
    if sims.size == 0:
        return did_video("No dispongo de esa información en mis registros.")

    idx = int(np.argmax(sims))

    # Umbral de similitud
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
  let t = document.getElementById('base').value;
  if(!t.trim()){ alert("Pega el texto primero"); return; }
  let r = await fetch('/load',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:t})
  });
  let j = await r.json();
  if(j.ok){
    document.getElementById('base').style.display='none';
    alert("Base cargada: " + j.pairs + " pares pregunta-respuesta");
  }
}

async function ask(){
  let q = document.getElementById('q').value;
  if(!q.trim()){ alert("Escribe una pregunta"); return; }
  let r = await fetch('/ask',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})
  });
  let j = await r.json();
  if(j.result_url){
    document.getElementById('v').src = j.result_url;
  } else {
    alert("No se pudo generar video");
    console.log(j);
  }
}
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return FRONT
