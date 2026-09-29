#!/usr/bin/env python3
"""Sustained int4 dot-product rate (v_dot8_i32_i4) of one gfx906, with the
shader clock sampled while it runs. The real ceiling for int4 GEMM kernels:
2026-09-29 on a Radeon VII, ~94 TOPS at 4 WGs/CU, not the 110 of 1.8 GHz x
spec, because sclk sags to 1.37-1.8 GHz under full load.

Usage:
    python develop/harness/peak_dot8.py --device cuda:1
"""
import argparse
import json
import os
import subprocess
import threading
import time

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_sclk(device_index, samples, stop):
    while not stop.is_set():
        smi = subprocess.run(["rocm-smi", "-d", str(device_index), "--showclocks"], capture_output=True, text=True, timeout=10)
        samples.extend(line.split("(")[-1].rstrip(")Mhz ") for line in smi.stdout.splitlines() if "sclk" in line)
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds", type=float, default=3.0, help="load duration per occupancy level")
    args = parser.parse_args()

    ext = load(name="peak_dot8", sources=[os.path.join(HERE, "peak_dot8.hip")],
               extra_cuda_cflags=["--offload-arch=gfx906", "-O3"], verbose=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    out = torch.empty(cus * 8 * 256, dtype=torch.int32, device=device)
    iters = 20000

    for blocks_per_cu in (2, 4, 8):
        blocks = cus * blocks_per_cu
        ext.run(out, blocks, 100)
        torch.cuda.synchronize()
        sclk, stop = [], threading.Event()
        sampler = threading.Thread(target=sample_sclk, args=(device.index, sclk, stop))
        sampler.start()
        ops, t0 = 0.0, time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            ops += ext.run(out, blocks, iters)
            # one launch in flight: queuing freely would outlast --seconds by minutes
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        stop.set()
        sampler.join()
        print(json.dumps({"cus": cus, "wgs_per_cu": blocks_per_cu, "tops": round(ops / elapsed / 1e12, 1),
                          "sclk_mhz": sclk[-4:]}), flush=True)


if __name__ == "__main__":
    main()
