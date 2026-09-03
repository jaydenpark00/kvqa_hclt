"""KVQA FINAL — review HTML (sanity-30 wiring page  +  representative/failure page).

  python -m src.gqa.kvqa_final_review --mode sanity30
  python -m src.gqa.kvqa_final_review --mode full
"""
from __future__ import annotations

import argparse
import base64
import glob
import io
import json

from src.common import read_jsonl, resolve
from src.gqa.kvqa_softscore import soft_accuracy


def img_uri(path, maxw=430, q=68):
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if im.width > maxw:
            im = im.resize((maxw, int(im.height * maxw / im.width)))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return ""


def esc(s):
    return str(s if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


CSS = ("<style>body{font-family:system-ui,Arial;margin:16px;background:#fafafa}"
       ".c{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;max-width:1040px}"
       "img{max-width:430px;border:1px solid #ccc;border-radius:6px}pre{white-space:pre-wrap;margin:3px 0;font-size:12.5px}"
       ".q{background:#f6f6ff;border-left:3px solid #55a;padding:5px 8px}"
       ".gen{background:#eef7ef;border-left:3px solid #3a7;padding:5px 8px}"
       ".hclt{background:#eef1fb;border-left:3px solid #46c;padding:5px 8px}"
       ".dec{background:#f3eefb;border-left:3px solid #85a;padding:5px 8px}"
       ".raw{background:#f0f0f0;border-left:3px solid #999;padding:5px 8px}"
       ".as{background:#fff4e8;border-left:3px solid #e90;padding:5px 8px}"
       ".fl{background:#fff0f0;border-left:3px solid #c55;padding:5px 8px;font-size:12px}"
       "table{border-collapse:collapse;font-size:12.5px;margin-top:6px}td{border:1px solid #e2e5ea;padding:3px 8px}</style>")


def _load(d, pat, tk, idk="id"):
    o = {}
    for fp in sorted(glob.glob(f"{d}/{pat}")):
        for r in read_jsonl(fp):
            if r.get(tk) is not None and r.get(idk):
                o[r[idk]] = r
    return o


def flagstr(lk):
    if not lk:
        return "(none)"
    return (f"NOVEL_GOLD={'Y' if lk.get('any_novel_gold_mention') else '·'}"
            f"(x{lk.get('novel_gold_match_count', 0)})  "
            f"RESULT_PROP={'Y' if lk.get('result_proposition') else '·'}  "
            f"CAND_ENUM={'Y' if lk.get('candidate_enumeration') else '·'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity30", "full"], required=True)
    ap.add_argument("--dir", default="outputs/kvqa_final_hclt_vs_ours_as_2k")
    ap.add_argument("--manifest", default="outputs/kvqa_final_hclt_vs_ours_as_2k/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    d = str(resolve(args.dir))
    man = {r["id"]: r for r in read_jsonl(resolve(args.manifest))}

    gen = _load(d, "generic_captions*.jsonl", "generic_caption")
    dec = _load(d, "ours_decompositions*.jsonl", "decomposition")
    rawv = _load(d, "ours_raw_evidence*.jsonl", "raw_evidence")
    asv = _load(d, "ours_suppressed_evidence*.jsonl", "suppressed_evidence")
    hcl = {r["id"]: r for r in read_jsonl(f"{d}/hclt_high_density.jsonl")} if glob.glob(f"{d}/hclt_high_density.jsonl") else {}
    leak = {name: (_load(d, f"leakage_{name}.jsonl", "present") if glob.glob(f"{d}/leakage_{name}.jsonl") else {})
            for name in ("generic", "hclt", "raw", "as")}
    CONDS = ["iq", "generic_text", "generic_mm", "hclt_text", "hclt_mm", "raw_text", "raw_mm", "as_text", "as_mm"]
    preds = {}
    for c in CONDS:
        m = {}
        for fp in sorted(glob.glob(f"{d}/predictions_{c}.jsonl") + glob.glob(f"{d}/predictions_{c}.shard*.jsonl")):
            for r in read_jsonl(fp):
                if "pred" in r:
                    m[r["id"]] = r["pred"]
        preds[c] = m

    ids = [i for i in man if i in gen and i in dec and i in rawv and i in asv][: args.limit] \
        if args.mode == "sanity30" else \
        sorted({i for lst in json.load(open(f"{d}/representative_examples.json")).values() for i in lst}) \
        if glob.glob(f"{d}/representative_examples.json") else list(man)[:60]

    title = f"KVQA FINAL — {args.mode} ({len(ids)} samples)"
    parts = [f"<title>{title}</title>", CSS, f"<h2>{title}</h2>"]
    for sid in ids:
        m = man[sid]
        parts.append('<div class="c">')
        parts.append(f'<div><b>{esc(sid)}</b> · {esc(m["answer_type"])} · answerable={m["answerable"]}</div>')
        u = img_uri(m["image_path"])
        if u:
            parts.append(f'<img src="{u}"/>')
        parts.append(f'<pre class="q"><b>Q:</b> {esc(m["question"])}\n<b>10 answers:</b> {esc(m["answers_10"])}</pre>')
        parts.append(f'<pre class="gen"><b>[GENERIC]</b>\n{esc(gen.get(sid, {}).get("generic_caption", "(missing)"))}</pre>')
        h = hcl.get(sid, {})
        if h:
            lines = []
            sel = h.get("selected_indices", [])
            for j, s in enumerate(h.get("sentences", [])):
                rk = str(sel.index(j) + 1) if j in sel else "·"
                sv = h["similarities"][j] if j < len(h.get("similarities", [])) else 0.0
                lines.append("  [%s] sim=%.3f  %s" % (rk, sv, esc(s)))
            parts.append('<pre class="hclt"><b>[HCLT] sentences (sim; [n]=selected rank)</b>\n' + "\n".join(lines)
                         + f'\n<b>final:</b> {esc(h.get("hclt_high_density_caption"))}</pre>')
        parts.append(f'<pre class="dec"><b>[OURS DECOMPOSITION]</b> (parse={esc(dec.get(sid, {}).get("parse_status"))})\n'
                     f'{esc(json.dumps(dec.get(sid, {}).get("decomposition", {}), ensure_ascii=False, indent=2))}</pre>')
        parts.append(f'<pre class="raw"><b>[OURS RAW]</b>\n{esc(rawv.get(sid, {}).get("raw_evidence", "(missing)"))}</pre>')
        parts.append(f'<pre class="as"><b>[OURS ANSWER-SUPPRESSED]</b>\n{esc(asv.get(sid, {}).get("suppressed_evidence", "(missing)"))}</pre>')
        parts.append('<pre class="fl"><b>[LEAKAGE]</b>\n'
                     f'Generic : {flagstr(leak["generic"].get(sid))}\nHCLT    : {flagstr(leak["hclt"].get(sid))}\n'
                     f'Ours-Raw: {flagstr(leak["raw"].get(sid))}\nOurs-AS : {flagstr(leak["as"].get(sid))}</pre>')
        trs = []
        for c in CONDS:
            p = preds[c].get(sid)
            if p is None:
                trs.append(f"<tr><td>{c}</td><td>(none)</td><td></td></tr>")
            else:
                s, kk, npd = soft_accuracy(p, m["answers_10"])
                trs.append(f"<tr><td>{c}</td><td>{esc(p)}</td><td>k={kk} soft={s:.3f}</td></tr>")
        parts.append("<table><tr><td><b>condition</b></td><td><b>raw pred</b></td><td><b>score</b></td></tr>"
                     + "".join(trs) + "</table></div>")

    out = f"{d}/{'sanity30_review.html' if args.mode == 'sanity30' else 'review.html'}"
    open(resolve(out), "w", encoding="utf-8").write("\n".join(parts))
    print(f"[kvqa-review] {args.mode}: wrote {out}  ({len(ids)} samples)")


if __name__ == "__main__":
    main()
