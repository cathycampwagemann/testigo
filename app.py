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


FRONT = r"""
<!doctype html><html><head><meta charset="utf-8"/>
<title>Testigo AI - Alessio</title>
<style>
body{
  font-family:sans-serif;
  background:#eef2f5;
  text-align:center;
  padding:20px;
}
.card{
  max-width:900px;
  margin:auto;
  background:#fff;
  padding:20px;
  border-radius:14px;
  box-shadow:0 4px 10px rgba(0,0,0,0.08);
}
.btn{
  padding:10px 14px;
  background:#111;
  color:#fff;
  border-radius:8px;
  cursor:pointer;
  margin:6px;
  border:none;
}
.btn:disabled{
  opacity:0.5;
  cursor:not-allowed;
}
textarea{
  width:90%;
  margin-top:10px;
}
#avatarWrap{
  position:relative;
  width:460px;
  max-width:100%;
  margin:20px auto;
}
#avatar{
  width:100%;
  border-radius:12px;
  background:#000;
}
#poster{
  position:absolute;
  inset:0;
  background:url('https://i.ibb.co/Tx4fBvDj/Alessio.jpg');
  background-size:cover;
  background-position:center;
  border-radius:12px;
}
</style></head><body>
<div class="card">

<h2>🎤 Testigo AI - Alessio</h2>

<!-- Botones principales -->
<button id="speak" class="btn">🎤 Hablar</button>
<button id="stop" class="btn">⏹️ Detener</button>
<button id="loadbtn" class="btn">📚 Cargar base RAG</button>

<!-- Base de interrogatorio -->
<div id="baseSection">
  <textarea id="base" rows="8" placeholder="Pega el texto completo del documento aquí"></textarea>
</div>

<!-- Pregunta -->
<textarea id="q" rows="3" placeholder="Escribe o dicta tu pregunta..."></textarea>
<button id="ask" class="btn">Preguntar</button>

<!-- Video del avatar -->
<div id="avatarWrap">
  <video id="avatar" autoplay playsinline controls></video>
  <div id="poster"></div>
</div>

</div>

<script>
const poster = document.getElementById('poster');
const video  = document.getElementById('avatar');
let rec = null;

// Ocultar / mostrar poster según el estado del video
video.addEventListener('play', ()=> poster.style.display='none');
video.addEventListener('ended',()=> poster.style.display='block');
video.addEventListener('pause',()=> {
  if(!video.currentTime) poster.style.display='block';
});

// Cargar base RAG
document.getElementById('loadbtn').onclick = async () => {
  let t = document.getElementById('base').value.trim();
  if(!t){
    alert("Pega el texto completo del documento primero.");
    return;
  }
  try{
    let r = await fetch('/load', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:t})
    });
    let j = await r.json();
    if(j.ok){
      if(j.pairs && j.pairs > 0){
        document.getElementById('baseSection').style.display='none';
        alert("✅ Base cargada. Pares pregunta-respuesta detectados: " + j.pairs);
      }else{
        alert("⚠️ Base cargada pero no se detectaron pares 'Para que...' / 'Respecto...'. Revisa el formato del texto.");
      }
    }else{
      alert("⚠️ No se pudo cargar la base.");
    }
  }catch(e){
    console.error(e);
    alert("Error llamando a /load");
  }
};

// Preguntar al testigo
document.getElementById('ask').onclick = async () => {
  let q = document.getElementById('q').value.trim();
  if(!q){
    alert("Escribe o dicta una pregunta primero.");
    return;
  }
  try{
    let r = await fetch('/ask', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})
    });
    let j = await r.json();
    if(j.result_url){
      video.src = j.result_url;
      video.play();
    }else{
      alert("⚠️ No se pudo generar respuesta en video.");
      console.log("Respuesta cruda /ask:", j);
    }
  }catch(e){
    console.error(e);
    alert("Error llamando a /ask");
  }
};

// Reconocimiento de voz (Chrome)
document.getElementById('speak').onclick = () => {
  if(!('webkitSpeechRecognition' in window)){
    alert("El dictado por voz sólo funciona en Chrome (desktop).");
    return;
  }
  if(rec){
    rec.stop();
  }
  rec = new webkitSpeechRecognition();
  rec.lang = 'es-ES';  // puedes cambiar a 'es-CL' en algunos navegadores
  rec.continuous = true;
  rec.interimResults = true;

  rec.onresult = (e) => {
    let texto = '';
    for(let i=0; i < e.results.length; i++){
      texto += e.results[i][0].transcript;
    }
    document.getElementById('q').value = texto;
  };

  rec.onerror = (e) => {
    console.error("Speech error:", e);
  };

  rec.start();
};

document.getElementById('stop').onclick = () => {
  if(rec){
    rec.stop();
    rec = null;
  }
};
</script>

</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return FRONT

