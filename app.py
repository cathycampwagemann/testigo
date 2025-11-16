from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import requests
import os
import time

# ====== CLAVES (en producción: usar variables de entorno en Render) ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "TU_CLAVE_AQUI_SI_ESTÁS_EN_LOCAL")
DID_API_KEY = os.getenv("DID_API_KEY", "TU_CLAVE_DID_AQUI_SI_ESTÁS_EN_LOCAL")
# ========================================================================

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

    # producto punto entre cada fila de M y q
    num = M @ q
    # norma de cada fila y del query
    M_norm = np.linalg.norm(M, axis=1)
    q_norm = np.linalg.norm(q)

    denom = (M_norm * q_norm) + 1e-10  # evitar división por cero
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


def did_video(text: str):
    data = {
        "script": {
            "type": "text",
            "provider": {"type": "microsoft", "voice_id": "es-US-AlonsoNeural"},
            "input": text.strip(),
        },
        "config": {
            "fluent": True,
            "pad_audio": 0.5,
        },
        "source_url": "https://i.ibb.co/Tx4fBvDj/Alessio.jpg",
    }

    headers = {"Authorization": f"Basic {DID_API_KEY}"}
    # crear el talk
    r = requests.post("https://api.d-id.com/talks", headers=headers, json=data).json()
