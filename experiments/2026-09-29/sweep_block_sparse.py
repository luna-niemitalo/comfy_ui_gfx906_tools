"""Calibrate on step 0, apply on later steps: tau x calibration blocks sweep.
Per (block, tau, n_calib): sparse heads, skipped fraction on the later step,
kernel time and error vs fp32 exact there."""
import importlib, json, os, sys, time
REPO = "/media/luna/work/tinker/ComfyUI"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "custom_nodes"))
import torch  # noqa: E402
att = importlib.import_module("ComfyUI-Gfx906Int4.attention")
bs = importlib.import_module("ComfyUI-Gfx906Int4.block_sparse")
dev = torch.device(sys.argv[1])
torch.cuda.set_device(dev)
CAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_capture_hr")
ROWS = att.block_rows("int4")
MAX_MISSED = 0.02


def time_ms(fn, iters=3):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def reference(q, k, v):
    H, L, D = q.shape[1:]
    rep = H // k.shape[1]
    out = torch.empty(1, L, H, D, device=dev)
    for h in range(H):
        for r0 in range(0, L, 2048):
            p = torch.softmax(q[0, h, r0:r0 + 2048].float() @ k[0, h // rep].float().T * D ** -0.5, -1)
            out[0, r0:r0 + 2048, h] = p @ v[0, h // rep].float()
    return out.reshape(1, L, H * D)


def rel(a, ref):
    return float(f"{((a.float() - ref).norm() / ref.norm()).item():.2e}")


def load(name):
    d = torch.load(os.path.join(CAP, name + ".pt"))
    return d["q"].to(dev), d["k"].to(dev), d["v"].to(dev)


for block in (0, 18, 27):
    qc, kc, _ = load(f"step0_block{block}")
    calib = {}
    for tau in (0.98, 0.99, 0.995):
        keep_c = bs.predict_keep(qc, kc, tau, ROWS)
        for n in (8, 16):
            bs.CALIBRATION_BLOCKS = n
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            calib[(tau, n)] = bs.missed_mass(qc, kc, keep_c, ROWS) <= MAX_MISSED
            torch.cuda.synchronize()
            calib[(tau, n, "ms")] = (time.perf_counter() - t0) * 1000
    del qc, kc
    for step in (2, 5):
        q, k, v = load(f"step{step}_block{block}")
        ref = reference(q, k, v)
        dense = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4)
        rec = {"block": block, "step": step, "dense_ms": round(time_ms(dense), 1), "dense_err": rel(dense(), ref)}
        for tau in (0.98, 0.99, 0.995):
            keep = bs.predict_keep(q, k, tau, ROWS)
            for n in (8, 16):
                sparse = calib[(tau, n)]
                final = (keep | ~sparse[None, :, None, None]).to(torch.uint8)
                masked = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4, final)
                rec[f"t{tau}_n{n}"] = {"heads": int(sparse.sum()), "skip": round(1 - final.float().mean().item(), 3),
                                       "ms": round(time_ms(masked), 1), "err": rel(masked(), ref),
                                       "calib_ms": round(calib[(tau, n, "ms")], 1)}
        print(json.dumps(rec), flush=True)
        del q, k, v, ref
        torch.cuda.empty_cache()
