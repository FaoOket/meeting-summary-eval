"""Backends branchables : ASR (Whisper) et résumé/juge (Qwen).

- En production (Colab GPU) : USE_MOCK=False, les vrais modèles tournent.
- En test local (pas de GPU) : USE_MOCK=True, on renvoie des sorties factices
  pour vérifier que la logique d'évaluation est correcte.

Le harnais evaluate.py n'appelle QUE les 3 fonctions publiques ci-dessous :
transcribe(), summarize(), judge_faithfulness(). Tu peux donc remplacer
Whisper/Qwen par n'importe quel autre modèle sans toucher au reste.
"""
import os
import time

USE_MOCK = os.environ.get("USE_MOCK", "0") == "1"

# --- Prompt de résumé (identique à ton PoC, centralisé ici) ---
SUMMARY_PROMPT = (
    "Tu es un assistant qui résume des réunions. À partir de la transcription "
    "ci-dessous, produis un compte-rendu clair en français avec EXACTEMENT ces "
    "trois sections :\n## Points clés\n## Décisions\n## Actions à faire (avec le responsable)\n\n"
    "N'invente rien : utilise uniquement ce qui est dit dans la transcription.\n\n"
    "Transcription :\n{transcript}"
)

JUDGE_PROMPT = (
    "Voici une transcription de réunion et un résumé. Le résumé est-il FIDÈLE "
    "(aucune information inventée, absente de la transcription) ?\n"
    "Réponds STRICTEMENT au format : 'NOTE: X/5' sur la première ligne "
    "(X entre 0 et 5), puis la liste des phrases inventées s'il y en a.\n\n"
    "TRANSCRIPTION:\n{transcript}\n\nRÉSUMÉ:\n{summary}"
)

_asr = None
_llm = None
_tok = None


def _load_models():
    """Charge Whisper + Qwen une seule fois (lazy)."""
    global _asr, _llm, _tok
    if _asr is not None:
        return
    from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline)
    _asr = pipeline("automatic-speech-recognition",
                    model="openai/whisper-small", device=0, chunk_length_s=30)
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    _tok = AutoTokenizer.from_pretrained(name)
    _llm = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype="auto", device_map="auto")


def transcribe(audio_path):
    """Audio -> (texte, latence_secondes)."""
    if USE_MOCK:
        time.sleep(0.01)
        return _MOCK_TRANSCRIPTS.get(
            os.path.basename(audio_path).split(".")[0], "transcription factice"), 0.01
    _load_models()
    t0 = time.time()
    text = _asr(audio_path)["text"].strip()
    return text, time.time() - t0


def _generate(prompt, max_new_tokens=400):
    inputs = _tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True).to(_llm.device)
    out = _llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return _tok.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True).strip()


def summarize(transcript):
    """Transcription -> (résumé, latence_secondes)."""
    if USE_MOCK:
        time.sleep(0.01)
        return _MOCK_SUMMARIES.get(transcript[:20], transcript), 0.01
    _load_models()
    t0 = time.time()
    summary = _generate(SUMMARY_PROMPT.format(transcript=transcript))
    return summary, time.time() - t0


def judge_faithfulness(transcript, summary):
    """LLM-as-judge -> note de fidélité /5 (float) + texte brut du juge."""
    if USE_MOCK:
        return 5.0, "NOTE: 5/5"
    _load_models()
    raw = _generate(JUDGE_PROMPT.format(transcript=transcript, summary=summary),
                    max_new_tokens=250)
    return _parse_note(raw), raw


def _parse_note(raw):
    """Extrait la note /5 de la réponse du juge de façon robuste."""
    import re
    m = re.search(r"(\d)(?:[.,]\d)?\s*/\s*5", raw)
    if m:
        return float(m.group(1))
    m = re.search(r"NOTE\s*[:=]\s*(\d)", raw, re.I)
    return float(m.group(1)) if m else float("nan")


# --- Données mock pour tester la logique d'évaluation hors GPU ---
_MOCK_TRANSCRIPTS = {}
_MOCK_SUMMARIES = {}
