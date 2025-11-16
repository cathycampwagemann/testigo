from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import requests
import os
import time

# ==========================
# CLAVES (usar variables de entorno en Render)
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "TU_CLAVE_OPENAI_LOCAL")
DID_API_KEY = os.getenv("DID_API_KEY", "TU_CLAVE_DID_LOCAL")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

QA_PAIRS = []
EMB_Q = None


# ==========================
# SIMILITUD COSENO SIN SKLEARN
# ==========================
def cosine_sim_vector_to_matrix(query_vec, matrix):
    q = np.array(query_vec, dtype=float)
    M = np.array(matrix, dtype=float)

    if M.size == 0:
        return np.array([])

    num = M @ q
    M_norm = np.linalg.norm(M, axis=1)
    q_norm = np.linalg.norm(q)

    denom = (M_norm * q_norm) + 1e-10
    return num / denom


# ==========================
# EMBEDDINGS
# ==========================
def embed(texts):
    vectors = []
    for t in texts:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        ).data[0].embedding
        vectors.append(emb)
    return np.array(vectors)


# ==========================
# D-ID VIDEO
# ==========================
def did_video(text):
    data = {
        "script": {
            "type": "text",
            "provider": {"type": "microsoft", "voice_id": "es-US-AlonsoNeural"},
            "input": text.strip()
        },
        "config": {
            "fluent": True,
            "pad_audio": 0.5
        },
        "source_url": "https://i.ibb.co/Tx4fBvDj/Alessio.jpg"
    }

    headers = {"Authorization": f"Basic {DID_API_KEY}"}
    res = requests.post("https://api.d-id.com/talks", headers=headers, json=data).json()

    if "id" not in res:
        return {"error": "Error al generar video", "raw": res}

    talk_id = res["id"]

    # Polling
    for _ in range(30):
        status = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers).json()
        if status.get("result_url"):
            return status
        time.sleep(1)

    return {"error": "timeout"}


# ==========================
# MODELOS Pydantic
# ==========================
class Load(BaseModel):
    text: str


class Ask(BaseModel):
    question: str


# ==========================
# ENDPOINT /load  (Carga base RAG)
# ==========================
@app.post("/load")
def load(b: Load):
    global QA_PAIRS, EMB_Q

    try:
        text = b.text.replace("\xa0", " ").replace("\r", " ").strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        QA_PAIRS = []

        # Detectar pares Pregunta -> Respuesta
        for i in range(len(lines) - 1):
            if ("Para que" in lines[i]) or ("Respecto" in lines[i]):
                QA_PAIRS.append((lines[i], lines[i + 1]))

        questions = [q for q, _ in QA_PAIRS]

        if questions:
            EMB_Q = embed(questions)
        else:
            EMB_Q = np.array([])

        return {"ok": True, "pairs": len(QA_PAIRS)}

    except Exception as e:
        print("ERROR en /load:", repr(e))
        return {"ok": False, "pairs": 0, "error": str(e)}


# ==========================
# ENDPOINT /ask (Pregunta al testigo)
# ==========================
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

    if sims[idx] < 0.40:
        return did_video("No dispongo de esa información en mis registros.")

    _, answer = QA_PAIRS[idx]
    return did_video(answer)


# ==========================
# FRONTEND HTML
# ==========================
FRONT = """
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
textarea{
  width:90%;
  margin-top:10px;
}
#avatarWrap{
  width:460px;
  max-width:100%;
  margin:20px auto;
  position:relative;
}
#avatar{
  width:100%;
  border-radius:12px;
}
#poster{
  position:absolute;
  inset:0;
  background:url('https://i.ibb.co/Tx4fBvDj/Alessio.jpg');
  background-size:cover;
  border-radius:12px;
}
</style></head>
<body>
<div class="card">

<h2>🎤 Testigo AI - Alessio</h2>

<button id="speak" class="btn">🎤 Hablar</button>
<button id="stop" class="btn">⏹️ Detener</button>
<button id="loadbtn" class="btn">📚 Cargar base RAG</button>

<div id="baseSection">
  <textarea id="base" rows="8" placeholder="Pega aquí el documento completo"></textarea>
</div>

<textarea id="q" rows="3" placeholder="Escribe o dicta tu pregunta..."></textarea>
<button id="ask" class="btn">Preguntar</button>

<div id="avatarWrap">
  <video id="avatar" autoplay playsinline controls></video>
  <div id="poster"></div>
</div>

</div>

<script>
const video = document.getElementById('avatar');
const poster = document.getElementById('poster');
let rec = null;

// Mostrar/ocultar poster
video.addEventListener('play',()=> poster.style.display='none');
video.addEventListener('ended',()=> poster.style.display='block');

// ====== Cargar base ======
document.getElementById('loadbtn').onclick = async () => {
  let t = document.getElementById('base').value.trim();
  if(!t){ alert("Pega el texto primero"); return; }

  try{
    let r = await fetch('/load',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:t})
    });
    let j = await r.json();

    if(j.ok){
      if(j.pairs > 0){
        document.getElementById('baseSection').style.display = 'none';
        document.getElementById('loadbtn').style.display = 'none';
        alert("Base cargada con éxito ("+j.pairs+" pares)");
      } else {
        alert("No se detectaron pares 'Para que...' o 'Respecto...'");
      }
    } else {
      alert("Error en /load: " + (j.error || "sin detalle"));
    }
  }catch(e){
    alert("Error llamando a /load: " + e);
  }
};

// ====== Preguntar ======
document.getElementById('ask').onclick = async () => {
  let q = document.getElementById('q').value.trim();
  if(!q){ alert("Haz una pregunta"); return; }

  try{
    let r = await fetch('/ask',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q})
    });
    let j = await r.json();

    if(j.result_url){
      video.src = j.result_url;
      video.play();
    } else {
      alert("No se pudo generar video");
    }
  }catch(e){
    alert("Error llamando a /ask: " + e);
  }
};

// ====== Dictado por voz ======
document.getElementById('speak').onclick = () => {
  if(!('webkitSpeechRecognition' in window)){
    alert("Usa Chrome para dictado por voz");
    return;
  }
  rec = new webkitSpeechRecognition();
  rec.lang = "es-ES";
  rec.continuous = true;
  rec.onresult = (e)=>{
    let txt = "";
    for(let i=0;i<e.results.length;i++){
      txt += e.results[i][0].transcript;
    }
    document.getElementById('q').value = txt;
  };
  rec.start();
};
document.getElementById('stop').onclick = ()=> rec && rec.stop();
</script>

</body></html>
"""


# ==========================
# ENDPOINTS FRONTEND
# ==========================
@app.get("/", response_class=HTMLResponse)
def home():
    return FRONT


@app.head("/")
def head_ok():
    return Response(status_code=200)
