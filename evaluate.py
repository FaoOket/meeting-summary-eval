"""Évaluation multi-réunions du pipeline audio -> Whisper -> Qwen -> résumé.

Lance sur TOUTES les réunions de meetings.json et agrège les métriques :
  - Couverture (%)      : part des points attendus retrouvés dans le résumé
  - WER (%)             : taux d'erreur de mots de l'ASR vs transcription de référence
  - Fidélité (/5)       : note LLM-as-judge (détection d'hallucinations)
  - Latence (s)         : temps ASR + résumé par réunion

Usage :
  python evaluate.py                 # vraie exécution (GPU requis)
  USE_MOCK=1 python evaluate.py      # test à blanc de la logique (sans GPU)
"""
import json
import statistics
import unicodedata
from pathlib import Path

import backends

ROOT = Path(__file__).resolve().parent


def normalize(text):
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")
    return text


def coverage(summary, expected):
    """Fraction des points attendus retrouvés dans le résumé."""
    s = normalize(summary)
    hits = []
    for item in expected:
        found = any(normalize(m) in s for m in item["any_of"])
        hits.append((item["label"], found))
    score = sum(f for _, f in hits) / len(hits) if hits else 0.0
    return score, hits


def word_error_rate(reference, hypothesis):
    """WER = (S + D + I) / N, via distance de Levenshtein au niveau des mots."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    n, m = len(ref), len(hyp)
    if n == 0:
        return float("nan")
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / n


def evaluate_meeting(meeting):
    audio = ROOT / meeting["audio"]
    transcript, lat_asr = backends.transcribe(str(audio))
    summary, lat_llm = backends.summarize(transcript)
    cov, hits = coverage(summary, meeting["expected"])
    faith, judge_raw = backends.judge_faithfulness(transcript, summary)

    wer = float("nan")
    ref_path = meeting.get("reference_transcript")
    if ref_path and (ROOT / ref_path).exists():
        wer = word_error_rate((ROOT / ref_path).read_text(encoding="utf-8"),
                              transcript)

    return {
        "id": meeting["id"],
        "duration_sec": meeting.get("duration_sec"),
        "coverage": round(cov, 3),
        "wer": round(wer, 3) if wer == wer else None,
        "faithfulness": faith if faith == faith else None,
        "latency_asr": round(lat_asr, 2),
        "latency_llm": round(lat_llm, 2),
        "latency_total": round(lat_asr + lat_llm, 2),
        "hits": [{"label": l, "found": f} for l, f in hits],
        "transcript": transcript,
        "summary": summary,
    }


def aggregate(rows):
    def clean(key):
        return [r[key] for r in rows if r.get(key) is not None]
    total_min = sum(r["duration_sec"] or 0 for r in rows) / 60
    covs = clean("coverage")
    wers = clean("wer")
    faiths = clean("faithfulness")
    lats = clean("latency_total")
    return {
        "n_reunions": len(rows),
        "duree_totale_min": round(total_min, 1),
        "couverture_moyenne": round(statistics.mean(covs), 3) if covs else None,
        "wer_moyen": round(statistics.mean(wers), 3) if wers else None,
        "fidelite_moyenne": round(statistics.mean(faiths), 2) if faiths else None,
        "hallucination_rate": round(sum(1 for f in faiths if f < 4) / len(faiths), 3) if faiths else None,
        "latence_mediane_s": round(statistics.median(lats), 1) if lats else None,
    }


def write_markdown(agg, rows, path):
    L = ["# Résultats d'évaluation — pipeline ASR + LLM\n"]
    L.append(f"**Jeu de test : {agg['n_reunions']} réunions, "
             f"{agg['duree_totale_min']} min d'audio au total.**\n")
    L.append("| Métrique | Valeur |")
    L.append("|---|---|")
    L.append(f"| Couverture moyenne des points clés | **{_pct(agg['couverture_moyenne'])}** |")
    L.append(f"| WER moyen (qualité transcription) | {_pct(agg['wer_moyen'])} |")
    L.append(f"| Fidélité moyenne (LLM-as-judge /5) | {agg['fidelite_moyenne']} |")
    L.append(f"| Taux d'hallucination (note < 4/5) | {_pct(agg['hallucination_rate'])} |")
    L.append(f"| Latence médiane par réunion | {agg['latence_mediane_s']} s |\n")
    L.append("## Détail par réunion\n")
    L.append("| Réunion | Durée | Couverture | WER | Fidélité | Latence |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['id']} | {r['duration_sec']}s | {_pct(r['coverage'])} | "
                 f"{_pct(r['wer'])} | {r['faithfulness']}/5 | {r['latency_total']}s |")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def _pct(x):
    return "—" if x is None else f"{x:.0%}"


def main():
    meetings = json.loads((ROOT / "meetings.json").read_text(encoding="utf-8"))["meetings"]
    rows = []
    for mtg in meetings:
        print(f"→ {mtg['id']} …")
        rows.append(evaluate_meeting(mtg))
    agg = aggregate(rows)

    (ROOT / "results.json").write_text(
        json.dumps({"aggregate": agg, "per_meeting": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(agg, rows, ROOT / "results.md")

    print("\n=== AGRÉGATS ===")
    for k, v in agg.items():
        print(f"  {k:22s}: {v}")
    print("\n✓ results.json et results.md générés.")


if __name__ == "__main__":
    main()
