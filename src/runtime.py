"""Process setup that MUST run before torch / CUDA is imported.

Call `setup_runtime(...)` as the very first thing in every GPU script:
  * sets the process title to "jaydenpark" (brief §2)
  * picks a free physical GPU and pins CUDA_VISIBLE_DEVICES to it (brief §1)
Inside the process the visible device is then always cuda:0.
"""
from __future__ import annotations

import os


def setup_runtime(prefer_gpu: int | None = None, proctitle: str = "jaydenpark", cpu_only: bool = False) -> dict:
    try:
        from setproctitle import setproctitle
        setproctitle(proctitle)
    except Exception as e:  # pragma: no cover
        print(f"[runtime] WARNING: could not set process title: {e}")

    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        info = {"proctitle": proctitle, "cpu_only": True, "physical_gpu": None, "visible_device": "cpu"}
        print(f"[runtime] {info}")
        return info

    forced = os.environ.get("GQA_FORCE_GPU")
    if forced not in (None, ""):
        # explicit pin for deliberate GPU packing (multiple small workers on one card).
        # caller asserts the GPU is theirs / shareable; skips the free-check.
        phys = int(forced)
        print(f"[runtime] GQA_FORCE_GPU set -> pinning physical GPU {phys} (free-check skipped)")
    else:
        from src.gpu_utils import pick_free_gpu
        phys = pick_free_gpu(prefer=prefer_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    info = {
        "proctitle": proctitle,
        "cpu_only": False,
        "physical_gpu": phys,
        "cuda_visible_devices": str(phys),
        "visible_device": "cuda:0",
    }
    print("[runtime] " + "\n[runtime] ".join([
        f"Selected physical GPU: {phys}",
        f"CUDA_VISIBLE_DEVICES={phys}",
        "Visible CUDA device inside process: cuda:0",
        f"Process title: {proctitle}",
    ]), flush=True)
    return info
