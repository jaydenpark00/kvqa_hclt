"""GPU selection for a shared server.

Rules (see project brief §1, §24):
  * only use a GPU with ~0 MiB memory in use AND no compute process
  * never pick a GPU another user is on; never kill anything
  * if nothing is clearly free, raise -- do NOT auto-fallback
Physical index is what we pass as CUDA_VISIBLE_DEVICES; inside the process that
device is always cuda:0.
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class GpuState:
    index: int
    mem_used_mib: int
    mem_total_mib: int
    util_pct: int
    procs: list  # list[(pid, process_name)]

    @property
    def is_free(self) -> bool:
        return self.mem_used_mib <= self.FREE_MEM_MIB and len(self.procs) == 0

    FREE_MEM_MIB = 200  # allow driver/ECC overhead only


def query_gpus() -> list[GpuState]:
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True)
    root = ET.fromstring(xml)
    out: list[GpuState] = []
    for i, gpu in enumerate(root.findall("gpu")):
        fb = gpu.find("fb_memory_usage")
        used = int(fb.find("used").text.split()[0])
        total = int(fb.find("total").text.split()[0])
        util_node = gpu.find("utilization/gpu_util")
        util = int(util_node.text.split()[0]) if util_node is not None and "N/A" not in util_node.text else -1
        procs = []
        pnode = gpu.find("processes")
        if pnode is not None:
            for p in pnode.findall("process_info"):
                pid = p.find("pid").text
                name = p.find("process_name").text
                procs.append((pid, name))
        out.append(GpuState(index=i, mem_used_mib=used, mem_total_mib=total,
                            util_pct=util, procs=procs))
    return out


def format_table(gpus: list[GpuState]) -> str:
    lines = [f"{'GPU':>3} {'mem_used':>10} {'mem_total':>10} {'util%':>6}  procs"]
    for g in gpus:
        pstr = ", ".join(f"{pid}:{name.split('/')[-1]}" for pid, name in g.procs) or "-"
        flag = "  <== FREE" if g.is_free else ""
        lines.append(f"{g.index:>3} {g.mem_used_mib:>10} {g.mem_total_mib:>10} {g.util_pct:>6}  {pstr}{flag}")
    return "\n".join(lines)


def pick_free_gpu(prefer: int | None = None) -> int:
    """Return a physical GPU index that is clearly free, else raise RuntimeError."""
    gpus = query_gpus()
    free = [g for g in gpus if g.is_free]
    print("[gpu_utils] current GPU state:\n" + format_table(gpus), flush=True)
    if not free:
        raise RuntimeError(
            "No clearly-free GPU (need <=200 MiB used and no compute process). "
            "Refusing to auto-fallback onto a shared GPU. Run CPU-only steps or retry later."
        )
    if prefer is not None:
        for g in free:
            if g.index == prefer:
                print(f"[gpu_utils] selected preferred free GPU {prefer}", flush=True)
                return prefer
        print(f"[gpu_utils] preferred GPU {prefer} not free; choosing another free GPU", flush=True)
    # lowest memory use, then lowest index
    chosen = sorted(free, key=lambda g: (g.mem_used_mib, g.index))[0].index
    print(f"[gpu_utils] selected free GPU {chosen}", flush=True)
    return chosen


if __name__ == "__main__":
    import sys
    gpus = query_gpus()
    print(format_table(gpus))
    try:
        print("\nwould select physical GPU:", pick_free_gpu())
    except RuntimeError as e:
        print("\n" + str(e))
        sys.exit(1)
