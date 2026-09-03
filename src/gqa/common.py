"""Korean GQA (AI Hub 71711) — shared helpers for the caption-bottleneck extension.

Schema was audited first (reports/korean_gqa_schema_audit.md); nothing here is guessed.
This module is READ-ONLY w.r.t. the dataset and does NOT import torch.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata

# --------------------------------------------------------------------------------------
# Paths (the dataset lives outside this project; we only ever read it)
# --------------------------------------------------------------------------------------
GQA_ROOT = "/ceph_data/jaydenpark/datasets/korean_gqa"

QA_JSON = {
    "train": os.path.join(GQA_ROOT, "train_qa.json"),
    "val": os.path.join(GQA_ROOT, "val_qa.json"),
}
SG_JSON = {
    "train": os.path.join(GQA_ROOT, "train_scene_graph.json"),
    "val": os.path.join(GQA_ROOT, "val_scene_graph.json"),
}
IMG_DIR = {
    "train": os.path.join(GQA_ROOT, "images/train"),
    "val": os.path.join(GQA_ROOT, "images/val"),
}
# NOTE: no 'test' split exists on disk (AI Hub ships train+val labels only).


def image_path(split: str, scene_graph_id: str) -> str:
    return os.path.join(IMG_DIR[split], f"{scene_graph_id}.jpg")


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_qa(split: str):
    """Yield (scene_graph_id, qa_item) for every QA_list entry in a split."""
    for rec in load_json(QA_JSON[split]):
        sgid = rec["Scene_Graph_ID"]
        for qd in rec.get("QA_list", []):
            yield sgid, qd


# --------------------------------------------------------------------------------------
# Conservative Korean answer normalization (metric §6 of the brief).
#   - Unicode NFC
#   - strip outer whitespace / quotes / brackets
#   - collapse internal whitespace
#   - drop a single trailing sentence-final period ('.' or '。')
# NO synonym mapping, NO stemming beyond the trailing period, NO LLM.
# --------------------------------------------------------------------------------------
_QUOTES = "\"'“”‘’「」『』()[]<>《》"


def normalize_ko(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).strip()
    s = s.strip(_QUOTES).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.。]\s*$", "", s)          # single trailing full stop only
    return s.strip()


def exact_match(pred: str, gold: str) -> bool:
    """Raw exact match — no normalization. Reported as the floor."""
    return str(pred).strip() == str(gold).strip()


def norm_match(pred: str, gold: str) -> bool:
    """Primary pilot metric: conservative-normalized exact match, single reference."""
    return normalize_ko(pred) == normalize_ko(gold) and normalize_ko(gold) != ""


def contains_match(pred: str, gold: str) -> bool:
    """Lenient signal ONLY (reported separately, never primary): normalized gold is a
    whitespace-insensitive substring of normalized pred, or vice-versa."""
    p = normalize_ko(pred).replace(" ", "")
    g = normalize_ko(gold).replace(" ", "")
    if not g or not p:
        return False
    return g in p or p in g


# --------------------------------------------------------------------------------------
# Deterministic question-family classifier.
#   Inputs: question_type (multi-label list), Korean question string,
#           #grounding-objects in annotations.question.
#   Uses NO gold answer and NO model output.
# --------------------------------------------------------------------------------------
_CMP_CUES = [
    "더 큰", "더 작은", "더 긴", "더 짧은", "더 높", "더 낮", "더 두꺼운", "더 얇은",
    "두께가 더", "크기가 더", "비교했을 때", "비교하면", "중 어떤", "중에서 더",
    "같은가", "같습니까", "같나요", "동일한가", "동일합니까",
]
_REL_LOCALIZE_CUES = [
    "왼쪽에 있는", "오른쪽에 있는", "옆에 있는", "위에 있는", "아래에 있는", "밑에 있는",
    "앞에 있는", "뒤에 있는", "사이에 있는", "근처에 있는", "위의", "안에 있는",
    "기준으로", "왼쪽에 무엇", "오른쪽에 무엇", "위에 무엇", "아래에 무엇",
]
_POSSESS_CUES = ["들고 있는", "입고 있는", "신고 있는", "쓰고 있는", "착용한", "쥐고 있는", "잡고 있는"]

FAMILY_GROUP = {
    "comp_comparison": "COMPOSITIONAL",
    "comp_relation": "COMPOSITIONAL",
    "comp_attr_via_relation": "COMPOSITIONAL",
    "direct_global": "DIRECT",
    "direct_category": "DIRECT",
    "direct_object_attribute": "DIRECT",
    "exclude_full_answer": "EXCLUDE",
    "other": "OTHER",
}


def classify_family(question_type, question_ko: str, n_grounding_objects: int,
                    answer_type: str) -> str:
    qt = set(question_type or [])
    q = question_ko or ""

    if answer_type == "full_answer":
        return "exclude_full_answer"

    has_cmp = any(c in q for c in _CMP_CUES)
    has_localize = any(c in q for c in _REL_LOCALIZE_CUES) or any(c in q for c in _POSSESS_CUES)

    # --- compositional ---
    if has_cmp:
        return "comp_comparison"
    if "Relation" in qt:
        # only 3 objects/image; a Relation tag always needs ≥2-object resolution
        return "comp_relation"
    if has_localize:
        # attribute/identity of a target picked out only via a relation to another object
        return "comp_attr_via_relation"

    # --- direct ---
    if "Global" in qt:
        return "direct_global"          # weather / brightness / indoor-outdoor
    if "Category" in qt:
        return "direct_category"        # "which object is a <category>"
    if qt and qt <= {"Object", "Attributes"}:
        return "direct_object_attribute"  # single-target recognition / colour / material / shape / state

    return "other"


def family_group(family: str) -> str:
    return FAMILY_GROUP.get(family, "OTHER")


# --------------------------------------------------------------------------------------
# Deterministic hashing for reproducible sampling (independent of dict / file order).
# --------------------------------------------------------------------------------------
def stable_hash(*parts: str) -> str:
    h = hashlib.sha1(("\x1f".join(str(p) for p in parts)).encode("utf-8"))
    return h.hexdigest()


def n_grounding_objects(qa_item: dict) -> int:
    return sum(1 for o in qa_item.get("annotations", {}).get("question", [])
              if o.get("object_id") is not None)


# --------------------------------------------------------------------------------------
# PHASE B — the 4 answer-generation prompts.
#
# The answer-generation instruction (final line) is IDENTICAL across all four conditions:
#     "정답만 간단히 출력하고 설명하지 마세요."
# This is the ONE deliberate change vs Exp1/Exp3, whose templates ended with
#     "정답만 짧게 작성하세요."
# Rationale: keep every condition's output-format directive byte-identical so Bq/B0/B1/B2
# stay comparable, and cut generative verbosity (KVQA Exp3 saw "이것은 X입니다." answers).
#
#  * B0 preamble == configs/exp1.yaml prompts.vqa_baseline preamble   (only the final line differs)
#  * B1 preamble == configs/exp1.yaml prompts.vqa_with_caption preamble (only the final line differs)
#  * Bq, B2     : new (Exp1 had no question-only / caption-only-no-image template),
#                 built parallel to B0 / B1 with the same final line.
# --------------------------------------------------------------------------------------
_ANSWER_INSTRUCTION = "정답만 간단히 출력하고 설명하지 마세요."

PROMPT_BQ = (
    "다음 질문에 답하세요.\n\n"
    "질문:\n{question}\n\n" + _ANSWER_INSTRUCTION
)
PROMPT_B0 = (
    "다음 이미지를 참고하여 질문에 답하세요.\n\n"
    "질문:\n{question}\n\n" + _ANSWER_INSTRUCTION
)
PROMPT_B1 = (
    "다음 이미지와 이미지 설명을 참고하여 질문에 답하세요.\n\n"
    "이미지 설명:\n{caption}\n\n"
    "질문:\n{question}\n\n" + _ANSWER_INSTRUCTION
)
PROMPT_B2 = (
    "다음 이미지 설명을 참고하여 질문에 답하세요.\n\n"
    "이미지 설명:\n{caption}\n\n"
    "질문:\n{question}\n\n" + _ANSWER_INSTRUCTION
)

# closed adjectival / brightness / weather answers -> conjugation stems that still count as
# the SAME word appearing in the caption (morphology, NOT synonymy). Used ONLY for the
# secondary 'lexical_present_loose' diagnostic, never for the primary label.
_ADJ_STEMS = {
    "크다": ["크", "큰"], "작다": ["작", "작은"], "길다": ["길", "긴"], "짧다": ["짧", "짧은"],
    "높다": ["높", "높은"], "낮다": ["낮", "낮은"], "두껍다": ["두꺼", "두꺼운"], "얇다": ["얇", "얇은"],
    "맑다": ["맑", "맑은"], "흐리다": ["흐리", "흐린"], "밝다": ["밝", "밝은"], "어둡다": ["어두", "어두운"],
}


def caption_repetition_score(text: str) -> float:
    """0 = no repetition, ->1 = highly degenerate. Fraction of word-trigrams that are
    duplicates. Used only to FLAG degenerate greedy-decoding loops for reporting — we do
    NOT change the decoding (Exp1/Exp3 parity)."""
    w = (text or "").split()
    if len(w) < 6:
        return 0.0
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return round(1.0 - len(set(tri)) / len(tri), 4)


def lexical_answer_present(answer_norm: str, caption: str) -> bool:
    """PRIMARY label: the conservative-normalized gold surface form appears verbatim
    (whitespace-insensitive) in the caption. Not semantic — call it *lexical* answer presence."""
    g = normalize_ko(answer_norm).replace(" ", "")
    cap = normalize_ko(caption).replace(" ", "")
    if not g or not cap:
        return False
    return g in cap


def lexical_answer_present_loose(answer_norm: str, caption: str) -> bool:
    """SECONDARY diagnostic: primary match OR, for the 12 closed adjectival answers, a
    conjugation stem of the same word. Reported alongside — never replaces the primary."""
    if lexical_answer_present(answer_norm, caption):
        return True
    cap = normalize_ko(caption).replace(" ", "")
    for stem in _ADJ_STEMS.get(normalize_ko(answer_norm), []):
        if stem and stem in cap:
            return True
    return False
