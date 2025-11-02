# ============================================================
# IA-Testigo con Avatar Alessio + Voz + RAG Estricto (No streaming)
# Ejecutar en Google Colab
# ============================================================

# Cloudflared (túnel sin ngrok)
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared

import os, re, time, asyncio, subprocess, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from uvicorn import Config, Server

# ------------ EDITAR ESTA LÍNEA ------------
OPENAI_API_KEY = "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"  # <--- pon tu clave
# -------------------------------------------

DID_API_KEY = "Y2F0eWNhbXBAZ21haWwuY29t:bdpnussf2aNNVCXyQVmyC"

client = OpenAI(api_key=OPENAI_API_KEY)
CORPUS = []
EMB = None

def chunk(text):
    # Dividir por saltos de línea dobles → mantiene las intervenciones completas
    partes = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 60]

    enriched = []
    for p in partes:
        p2 = (p.replace("87%", "0.87")
                .replace("0,87", "87%")
                .replace("0.87", "87%")
                .replace("explicar", "contribución explicación modelo importancia variable feature"))
        enriched.append(p + "\n" + p2)

    return enriched

def embed(chunks):
    vec=[]
    for c in chunks:
        vec.append(client.embeddings.create(model="text-embedding-3-small", input=c).data[0].embedding)
    return np.array(vec)

def retrieve(q, k=5):
    global EMB, CORPUS
    qe = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    sims = cosine_similarity([qe], EMB)[0]
    idx = np.argsort(sims)[-k:][::-1]

    # Un solo bloque continuo (clave)
    contexto = "\n\n".join([CORPUS[i] for i in idx])
    return contexto

def answer(q, ctx):
    prompt = f"""
Responde únicamente usando el siguiente registro de interrogatorio (RAG estricto).
Si la respuesta no está explícita en el texto, responde exactamente:
"No dispongo de esa información en mis registros."

Registro (extractos relevantes):
{ctx}

Pregunta:
{q}

Instrucciones de estilo:
- Contesta como IA-testigo en audiencia.
- Usa tono formal y objetivo.
- Si la respuesta está en el texto, explica en pasos claros qué variables contribuyeron.
- Si el texto reconoce límites (por ejemplo, trazabilidad parcial), dilo textualmente.

Respuesta:
"""

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return r.choices[0].message.content.strip()

def did_video(text):
    import requests, time

    # 1) Crear la solicitud de habla
    create_payload = {
        "script": {
            "type": "text",
            "provider": {"type": "microsoft", "voice_id": "es-ES-AlvaroNeural"},
            "input": text.strip()
        },
        "config": {"fluent": True, "pad_audio": 0.5},
        "source_url": "https://i.ibb.co/Tx4fBvDj/Alessio.jpg"
    }

    headers = {"Authorization": f"Basic {DID_API_KEY}"}
    create_res = requests.post("https://api.d-id.com/talks", headers=headers, json=create_payload).json()

    # Si falló al crear:
    if "id" not in create_res:
        return {"error": "No se pudo crear el video", "raw": create_res}

    talk_id = create_res["id"]

    # 2) Polling hasta que el video esté listo:
    for _ in range(30):  # ~30 x 1s = 30 segundos máximo
        time.sleep(1)
        status = requests.get(f"https://api.d-id.com/talks/{talk_id}", headers=headers).json()

        if "result_url" in status and status["result_url"]:
            return {"result_url": status["result_url"]}

    return {"error": "timeout"}

app = FastAPI()
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

class Load(BaseModel):
    text:str
class Ask(BaseModel):
    question:str

@app.post("/load")
def load(b: Load):
    global QA_PAIRS, EMB_Q

    # 1) Normalizar texto
    text = b.text.replace("\xa0", " ").replace("\r", " ").strip()

    # 2) Detectar pares Pregunta → Respuesta
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    QA_PAIRS = []

    for i in range(len(lines)-1):
        if "Para que" in lines[i] or "Para que diga" in lines[i] or "Respecto" in lines[i]:
            question = lines[i]
            answer = lines[i+1] if i+1 < len(lines) else ""
            QA_PAIRS.append((question, answer))

    # 3) Embed solo las preguntas
    questions_only = [q for q, a in QA_PAIRS]
    EMB_Q = embed(questions_only)

    return {"ok": True, "pairs": len(QA_PAIRS)}

@app.post("/ask")
def ask(b: Ask):
    global QA_PAIRS, EMB_Q

    qe = client.embeddings.create(model="text-embedding-3-small", input=b.question).data[0].embedding
    sims = cosine_similarity([qe], EMB_Q)[0]
    idx = int(np.argmax(sims))  # La pregunta más parecida

    original_answer = QA_PAIRS[idx][1]

    # Si la pregunta no supera umbral mínimo → entonces sí decimos "no tengo info"
    if sims[idx] < 0.45:
        return did_video("No dispongo de esa información en mis registros.")

    return did_video(original_answer)

FRONT = r"""
<!doctype html><html><head><meta charset="utf-8"/>
<title>Testigo Avatar</title>
<style>
body{font-family:sans-serif;background:#eef2f5;text-align:center;padding:20px}
.card{max-width:800px;margin:auto;background:#fff;padding:20px;border-radius:14px}
.btn{padding:10px 14px;background:#111;color:#fff;border-radius:8px;cursor:pointer;margin:6px}
textarea{width:90%;margin-top:10px}
#avatarWrap{position:relative;width:460px;margin:auto}
#avatar{width:100%;border-radius:12px;background:#000}
#poster{position:absolute;inset:0;background:url('https://i.ibb.co/Tx4fBvDj/Alessio.jpg');background-size:cover;border-radius:12px}
</style></head><body>
<div class="card">

<button id="speak" class="btn">🎤 Hablar</button>
<button id="stop" class="btn">⏹️</button>
<button id="loadbtn" class="btn">📚 Cargar base RAG</button>

<div id="baseSection">
<textarea id="base" rows="8" placeholder="Pega el texto completo del documento aquí"></textarea>
</div>

<textarea id="q" rows="3" placeholder="Pregunta..."></textarea>
<button id="ask" class="btn">Preguntar</button>

<div id="avatarWrap">
<video id="avatar" autoplay playsinline controls></video>
<div id="poster"></div>
</div>

</div>

<script>
const poster=document.getElementById('poster');
const video=document.getElementById('avatar');
video.addEventListener('play',()=>poster.style.display='none');
video.addEventListener('ended',()=>poster.style.display='block');
video.addEventListener('pause',()=>poster.style.display=video.currentTime?'none':'block');

document.getElementById('loadbtn').onclick=async()=>{
 let t=document.getElementById('base').value.trim();
 if(!t){alert("Pega el texto primero");return;}
 let r=await fetch('/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
 let j=await r.json();
 if(j.ok){document.getElementById('baseSection').style.display='none';alert("✅ Base cargada");}
};

document.getElementById('ask').onclick=async()=>{
 let q=document.getElementById('q').value.trim();
 if(!q){alert("Haz una pregunta");return;}
 let r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 let j=await r.json();
 if(j.result_url){video.src=j.result_url;video.play();}
 else alert("⚠️ No se pudo generar respuesta");
};

let rec;
document.getElementById('speak').onclick=()=>{
 if(!('webkitSpeechRecognition' in window)){alert("Usa Chrome");return;}
 rec=new webkitSpeechRecognition();rec.lang='es-ES';rec.continuous=true;
 rec.onresult=e=>document.getElementById('q').value=[...e.results].map(r=>r[0].transcript).join(' ');
 rec.start();
};
document.getElementById('stop').onclick=()=>rec&&rec.stop();
</script>

</body></html>
"""

if __name__ == "__main__":
    # Modo: ejecutar directamente con python app.py
    # Para producción es mejor usar: uvicorn app:app --host 0.0.0.0 --port 8000
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")


