"""KVQA FINAL — HCLT-style High-relevance + High-density (ONE pipeline).  NO VLM, NO gold.

generic caption -> deterministic sentence split -> MiniLM(question, sentence) cosine ->
sort by similarity DESC (tie: earlier sentence) -> take Top-3 -> concat in that order.

Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 via transformers
AutoModel + mean pooling + L2 normalize (== sentence-transformers for this model).

  python -m src.gqa.kvqa_final_hclt
"""
from __future__ import annotations

import argparse
import glob
import re

from src.runtime import setup_runtime

_SPLIT = re.compile(r"(?<=[.?!。])\s+|\n+")
_LEAD_MARK = re.compile(r"^\s*(?:[-*•·]|\d+[.)]|[가-힣]\.)\s+")
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def split_sentences(text: str) -> list[str]:
    out = []
    for piece in _SPLIT.split(text or ""):
        s = _LEAD_MARK.sub("", piece.strip()).strip().strip("`*_ ").strip()
        if len(s) >= 2:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--prefer-gpu", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    setup_runtime(prefer_gpu=args.prefer_gpu)

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from src.common import read_jsonl, JsonlWriter, resolve

    man = {r["id"]: r for r in read_jsonl(resolve(args.manifest))}
    gc = {}
    for fp in sorted(glob.glob(str(resolve(f"{args.outdir}/generic_captions*.jsonl")))):
        for r in read_jsonl(fp):
            if r.get("generic_caption"):
                gc[r["id"]] = r["generic_caption"]
    ids = [i for i in man if i in gc]
    print(f"[kvqa-hclt] {len(ids)}/{len(man)} generic captions present; model={EMB_MODEL}")

    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(dev).eval()

    @torch.inference_mode()
    def embed(texts):
        vecs = []
        for i in range(0, len(texts), args.batch):
            b = texts[i:i + args.batch]
            enc = tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            mean = torch.nn.functional.normalize(mean, p=2, dim=1)
            vecs.append(mean.cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, mdl.config.hidden_size), np.float32)

    sent_map = {i: split_sentences(gc[i]) for i in ids}
    q_emb = embed([man[i]["question"] for i in ids])

    w_sent = JsonlWriter(f"{args.outdir}/generic_sentences.jsonl")
    w = JsonlWriter(f"{args.outdir}/hclt_high_density.jsonl")
    n_deg = 0
    for k, i in enumerate(ids):
        sents = sent_map[i]
        w_sent.write({"id": i, "question": man[i]["question"], "n_sentences": len(sents), "sentences": sents})
        if not sents:
            w.write({"sample_id": i, "id": i, "question": man[i]["question"],
                     "answers_10": man[i]["answers_10"], "generic_caption": gc[i], "sentences": [],
                     "similarities": [], "ranked_indices": [], "ranked_sentences": [],
                     "selected_indices": [], "selected_sentences": [], "hclt_high_density_caption": ""})
            continue
        s_emb = embed(sents)
        sims = (s_emb @ q_emb[k]).tolist()
        ranked = sorted(range(len(sents)), key=lambda j: (-sims[j], j))   # relevance desc; tie -> earlier
        sel = ranked[: args.top_k]
        if len(sents) < args.top_k:
            n_deg += 1
        caption = " ".join(sents[j] for j in sel)
        w.write({"sample_id": i, "id": i, "question": man[i]["question"],
                 "answers_10": man[i]["answers_10"], "generic_caption": gc[i],
                 "sentences": sents, "similarities": [round(x, 6) for x in sims],
                 "ranked_indices": ranked, "ranked_sentences": [sents[j] for j in ranked],
                 "selected_indices": sel, "selected_sentences": [sents[j] for j in sel],
                 "hclt_high_density_caption": caption})
        if (k + 1) % 200 == 0 or k + 1 == len(ids):
            print(f"[kvqa-hclt] {k+1}/{len(ids)}", flush=True)
    w_sent.close(); w.close()
    n_sent = [len(sent_map[i]) for i in ids]
    print(f"[kvqa-hclt] done. avg sentences/caption {sum(n_sent)/max(len(n_sent),1):.1f}; "
          f"<{args.top_k} sentences in {n_deg}/{len(ids)} ({n_deg/max(len(ids),1):.1%})")


if __name__ == "__main__":
    main()
