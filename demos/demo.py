"""Demo: measured busbw vs the 2(N-1)/N formula, ring sizes 2..8. Receipts."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ring  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "receipts")
os.makedirs(OUT, exist_ok=True)

MEASURE = []
for N in (2, 3, 4, 5, 8):
    w = ring.init(N)
    try:
        x = np.random.default_rng(N).uniform(-1, 1, 2 ** 18)   # 2 MiB fp64
        t0 = time.perf_counter()
        res = ring.allreduce(x, w=w)
        dt = time.perf_counter() - t0
        assert all(np.allclose(r, N * x) for r in res)
        rep = ring.busbw(x.nbytes, dt, N)
        rep["N"] = N
        rep["seconds"] = round(dt, 4)
        MEASURE.append(rep)
    finally:
        ring.shutdown()

receipt = {
    "tensor": "2 MiB fp64, allreduce, loopback TCP, 16MB socket buffers",
    "measurements": MEASURE,
    "formula": "busbw = algbw * 2(N-1)/N ; wire bytes/rank = 2(N-1)/N * S",
    "note": "Single-shot local TCP timings include Python threads, pickle and copies. "
            "busbw is formula-normalized throughput, not measured wire bandwidth "
            "or evidence of NCCL/GPU scaling. Requested socket buffers may be OS-capped.",
}
with open(os.path.join(OUT, "receipt.json"), "w") as f:
    json.dump(receipt, f, indent=2)
print(json.dumps(receipt, indent=2))
assert len(MEASURE) == 5
assert MEASURE[-1]["busbw_gbps"] > 0
print("DEMO_OK")
