"""block_sparse.py on 2560x1440 captures: calibration verdict, skipped
fraction, kernel time and error vs fp32 exact (dense vs calibrated sparse)."""
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
TAU, MAX_MISSED = 0.995, 0.02


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


for name in sys.argv[2:]:
    d = torch.load(os.path.join(CAP, name + ".pt"))
    q, k, v = d["q"].to(dev), d["k"].to(dev), d["v"].to(dev)
    ref = reference(q, k, v)
    predict_ms = time_ms(lambda: bs.predict_keep(q, k, TAU, ROWS))
    keep = bs.predict_keep(q, k, TAU, ROWS)
    t0 = time.perf_counter()
    missed = bs.missed_mass(q, k, keep, ROWS)
    torch.cuda.synchronize()
    calib_ms = (time.perf_counter() - t0) * 1000
    sparse = missed <= MAX_MISSED
    final = (keep | ~sparse[None, :, None, None]).to(torch.uint8)
    dense = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4)
    masked = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4, final)
    print(json.dumps({"capture": name, "sparse_heads": int(sparse.sum()), "heads": len(sparse),
                      "missed_min_med_max": [round(x, 4) for x in (missed.min().item(), missed.median().item(), missed.max().item())],
                      "skipped": round(1 - final.float().mean().item(), 3),
                      "predict_ms": round(predict_ms, 1), "calib_ms": round(calib_ms, 1),
                      "dense_ms": round(time_ms(dense), 1), "sparse_ms": round(time_ms(masked), 1),
                      "dense_err": rel(dense(), ref), "sparse_err": rel(masked(), ref)}), flush=True)
    del q, k, v, ref
    torch.cuda.empty_cache()
