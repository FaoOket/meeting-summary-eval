"""
app.py — Interface de démonstration (Streamlit)
Pipeline : audio -> transcription (Whisper) -> résumé structuré (LLM Qwen).

Lancer en local :   streamlit run app.py
(Sur un PC sans GPU, garde WHISPER_SIZE="base" et le modèle LLM 1.5B : c'est plus lent
mais ça fonctionne. Sur Colab avec GPU, tu peux monter à "small".)
"""

import time
import streamlit as st
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

WHISPER_SIZE = "base"                       # "base" (CPU) -> "small" (GPU)
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"     # petit LLM, tient sur GPU gratuit / CPU

st.set_page_config(page_title="Résumé de réunion — PoC", page_icon="🎙️")
st.title("🎙️ Transcription & résumé automatique de réunion")
st.caption("Audio → transcription (Whisper) → résumé structuré (LLM). Démo / PoC.")


@st.cache_resource(show_spinner="Chargement du modèle de transcription…")
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(WHISPER_SIZE, device=device, compute_type=compute)


@st.cache_resource(show_spinner="Chargement du modèle de résumé…")
def load_llm():
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, torch_dtype="auto", device_map="auto"
    )
    return tok, model


def transcribe(path):
    model = load_whisper()
    segments, info = model.transcribe(path, language="fr")
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip(), info.language


def summarize(transcript):
    tok, model = load_llm()
    prompt = (
        "Tu es un assistant qui résume des réunions. À partir de la transcription "
        "ci-dessous, produis un compte-rendu clair en français avec EXACTEMENT ces "
        "trois sections :\n"
        "## Points clés\n## Décisions\n## Actions à faire (avec le responsable)\n\n"
        "N'invente rien : utilise uniquement ce qui est dit dans la transcription.\n\n"
        f"Transcription :\n{transcript}"
    )
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(inputs, max_new_tokens=400, do_sample=False)
    answer = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
    return answer.strip()


uploaded = st.file_uploader(
    "Dépose un fichier audio de réunion", type=["m4a", "mp3", "wav", "ogg"]
)

if uploaded:
    with open("uploaded_audio", "wb") as f:
        f.write(uploaded.read())

    t0 = time.time()
    with st.spinner("Transcription en cours…"):
        transcript, lang = transcribe("uploaded_audio")
    t1 = time.time()
    with st.spinner("Génération du résumé…"):
        summary = summarize(transcript)
    t2 = time.time()

    col1, col2 = st.columns(2)
    col1.metric("Transcription", f"{t1 - t0:.1f} s")
    col2.metric("Résumé", f"{t2 - t1:.1f} s")

    st.subheader("📝 Résumé")
    st.markdown(summary)

    with st.expander("Voir la transcription brute"):
        st.write(transcript)
