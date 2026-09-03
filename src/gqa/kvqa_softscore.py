"""KVQA soft VQA accuracy — VERBATIM reuse of the frozen HCLT-repo evaluator.

Source of truth (copied byte-for-byte, provenance kept):
  /ceph_data/jaydenpark/2026_HCLT-main/2026_HCLT-main/src/evaluate.py
  -> `normalize_answer` (SKTBrain/BAN-KVQA tools/compute_softscore.py punctuation processing)
  -> `soft_accuracy`   (min(exact normalized-match count / 3, 1))

NO semantic Korean rewrites, NO synonym dict, NO LLM judge, NO embedding match, NO translation.
kvqa_final_analyze runs `selftest()` (section 11) and cross-checks against the repo module
before any 2K scoring.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# --- verbatim from the HCLT repo src/evaluate.py -------------------------------------------
# From SKTBrain/BAN-KVQA tools/compute_softscore.py.
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCTUATION = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]

NORMALIZATION_NAME = "SKTBrain/BAN-KVQA punctuation + whitespace trim/collapse"
CATEGORY_ORDER = ("yes/no", "number", "other", "unanswerable")
CATEGORY_LABELS = {
    "yes/no": "Yes/No", "number": "Number", "other": "Other", "unanswerable": "Unanswerable",
}


def normalize_answer(answer: str) -> str:
    """Apply official punctuation processing without semantic Korean rewrites."""
    text = str(answer)
    original = text
    for punctuation in _PUNCTUATION:
        if (
            f"{punctuation} " in original
            or f" {punctuation}" in original
            or _COMMA_STRIP.search(original) is not None
        ):
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = _PERIOD_STRIP.sub("", text)
    text = text.replace(",", "")
    return " ".join(text.strip().split())


def soft_accuracy(prediction: str, gt_answers: Iterable[str]) -> tuple[float, int, str]:
    """Requested pilot metric: min(number of exact normalized matches / 3, 1)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_gt = [normalize_answer(answer) for answer in gt_answers]
    matches = Counter(normalized_gt)[normalized_prediction]
    return min(matches / 3.0, 1.0), matches, normalized_prediction
# -----------------------------------------------------------------------------------------

import os
_REPO = os.environ.get("HCLT_REPO", "")   # optional: path to the KVQA structured-caption repo for evaluator cross-check


def _repo_module():
    """Import the repo's src/evaluate.py in isolation (avoids the `src` package name clash)."""
    import importlib.util
    import os
    p = os.path.join(_REPO, "src", "evaluate.py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("_hclt_repo_evaluate", p)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules.setdefault("src", type(sys)("src"))  # satisfy `from src.utils import ...`
    try:
        import types
        u = types.ModuleType("src.utils")
        u.atomic_write_json = lambda *a, **k: None
        sys.modules["src.utils"] = u
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # pragma: no cover
        print(f"[kvqa-softscore] repo cross-check skipped: {e!r}")
        return None


def selftest() -> None:
    """Section 11 unit test — must pass before any 2K scoring."""
    # k -> expected soft score
    cases = {0: 0.0, 1: 1 / 3, 2: 2 / 3, 3: 1.0, 4: 1.0, 10: 1.0}
    for k, want in cases.items():
        gts = ["고양이"] * k + ["개"] * (10 - k)
        got, matches, _ = soft_accuracy("고양이", gts)
        assert matches == k, f"match count k={k}: got {matches}"
        assert abs(got - want) < 1e-9, f"k={k}: soft {got} != {want}"
    # normalization: punctuation stripped, whitespace collapsed
    assert normalize_answer(" 검정색.  ") == "검정색"
    assert normalize_answer("고양이?") == "고양이"
    s, m, _ = soft_accuracy("검정색", ["검정색", "검정색.", "  검정색 ", "까망", "검정색"])
    assert m == 4 and s == 1.0, (m, s)
    # 0 matches
    s, m, _ = soft_accuracy("빨강", ["파랑"] * 10)
    assert m == 0 and s == 0.0
    # cross-check against the repo module if importable
    rm = _repo_module()
    if rm is not None:
        import random
        rng = random.Random(11)
        vocab = ["검정색", "흰색", "고양이 두 마리", "3", "unanswerable", "예", "아니오", "왼쪽"]
        for _ in range(200):
            pred = rng.choice(vocab)
            gts = [rng.choice(vocab) for _ in range(10)]
            a = soft_accuracy(pred, gts)
            b = rm.soft_accuracy(pred, gts)
            assert a == b, f"repo mismatch pred={pred!r}: local {a} vs repo {b}"
        assert normalize_answer("a; b, 1,000") == rm.normalize_answer("a; b, 1,000")
        print("[kvqa-softscore] selftest PASS (+ repo cross-check on 200 random cases)")
    else:
        print("[kvqa-softscore] selftest PASS (repo module not importable; local copy verified)")


if __name__ == "__main__":
    selftest()
