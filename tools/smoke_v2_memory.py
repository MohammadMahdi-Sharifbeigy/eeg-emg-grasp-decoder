"""GPU memory and timing check for KG-GT v2 on random data.
Run from the repo root with the GPU free:  env/Scripts/python.exe tools/smoke_v2_memory.py"""
import sys

import torch

sys.path.insert(0, ".")
from v2.config import CONFIG_V2
from v2.smoke import project_runtime

if not torch.cuda.is_available():
    sys.exit("CUDA not available; nothing to measure.")
free, total = torch.cuda.mem_get_info()
print(f"GPU free {free / 2**30:.2f} GB of {total / 2**30:.2f} GB")
if free < 0.75 * total:
    sys.exit("The GPU is busy (less than 75 % free). Stop and tell the controller.")
out = project_runtime(CONFIG_V2, torch.device("cuda"))
print("measured settings (mixed precision, batch, accumulation, peak GB, s per step, projected minutes, fits):")
for c in out["candidates"]:
    if c["peak_gb"] is None:
        numbers = "out of memory"
    else:
        numbers = f"{c['peak_gb']:8.3f} {c['sec_per_step']:8.4f} {c['projected_minutes']:8.1f}"
    print(f"  {'on' if c['use_amp'] else 'off':4s} {c['batch_size']:4d} {c['gradient_accumulation_steps']:3d} "
          f"{numbers}  {'fits' if c['fits'] else 'does not fit'}")
print("chosen setting:")
for k, v in out.items():
    if k != "candidates":
        print(f"{k:28s} {v}")
