"""Block-sparse kernel with an oracle keep mask on 2560x1440 captures:
kernel time dense vs masked, and error vs fp32 exact attention (dense int4/u8,
masked int4/u8, and the fp32 masked emulation for the same mask)."""
import importlib, json, os, sys, time
REPO = "/media/luna/work/tinker/ComfyUI"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "custom_nodes"))
import torch  # noqa: E402
att = importlib.import_module("ComfyUI-Gfx906Int4.attention")
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
torch.cuda.set_device(dev)
CAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attn_capture_hr")
BQ, BK, ROWS = att.block_rows("int4"), 64, 2048


def time_ms(fn, iters=3):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def exact_and_mass(q, k, v, rep):
    """fp32 reference [H, L, D] and block mass [H, nqb, nkt]."""
    H, L, D = q.shape
    nkt, nqb = -(-L // BK), -(-L // BQ)
    ref = torch.empty(H, L, D, device=dev)
    mass = torch.zeros(H, nqb, nkt, device=dev)
    for h in range(H):
        kh, vh = k[h // rep].float(), v[h // rep].float()
        for r0 in range(0, L, ROWS):
            p = torch.softmax(q[h, r0:r0 + ROWS].float() @ kh.T * D ** -0.5, -1)
            ref[h, r0:r0 + ROWS] = p @ vh
            n = p.shape[0]
            pp = torch.nn.functional.pad(p, (0, nkt * BK - L, 0, -n % BQ))
            mass[h, r0 // BQ:r0 // BQ + pp.shape[0] // BQ] = pp.reshape(-1, BQ, nkt, BK).sum((1, 3))
    return ref, mass


def top_cdf(mass, tau):
    srt, idx = mass.sort(-1, descending=True)
    tot = srt.sum(-1, keepdim=True)
    keep_sorted = (srt.cumsum(-1) - srt) / tot < tau
    return torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)


def masked_emulation(q, k, v, rep, keep):
    H, L, D = q.shape
    out = torch.empty(H, L, D, device=dev)
    for h in range(H):
        kh, vh = k[h // rep].float(), v[h // rep].float()
        for r0 in range(0, L, ROWS):
            p = torch.softmax(q[h, r0:r0 + ROWS].float() @ kh.T * D ** -0.5, -1)
            km = keep[h, r0 // BQ:r0 // BQ + -(-p.shape[0] // BQ)].repeat_interleave(BQ, 0)[:p.shape[0]]
            pm = p * km.repeat_interleave(BK, 1)[:, :L]
            out[h, r0:r0 + ROWS] = pm @ vh / pm.sum(-1, keepdim=True)
    return out


def rel(a, ref):
    return float(f"{((a.float() - ref).norm() / ref.norm()).item():.2e}")


for name in sys.argv[2:]:
    d = torch.load(os.path.join(CAP, name + ".pt"))
    q, k, v = d["q"].to(dev), d["k"].to(dev), d["v"].to(dev)
    rep = q.shape[1] // k.shape[1]
    ref, mass = exact_and_mass(q[0], k[0], v[0], rep)
    ref_out = ref.permute(1, 0, 2).reshape(1, q.shape[2], -1)
    dense = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4)
    rec = {"capture": name, "L": q.shape[2], "dense_ms": round(time_ms(dense), 1), "dense_err": rel(dense(), ref_out)}
    for tau in (0.98, 0.995):
        keep = top_cdf(mass, tau)
        keep_u8 = keep.to(torch.uint8)[None].contiguous()
        sparse = lambda: att.fused_attention(q, k, v, "int4", "u8", 1e-4, keep_u8)
        emu = masked_emulation(q[0], k[0], v[0], rep, keep).permute(1, 0, 2).reshape(1, q.shape[2], -1)
        rec[f"tau{tau}"] = {"sparsity": round(1 - keep.float().mean().item(), 3), "ms": round(time_ms(sparse), 1),
                            "err": rel(sparse(), ref_out), "emu_err": rel(emu, ref_out)}
    print(json.dumps(rec), flush=True)
    del q, k, v, ref, mass
    torch.cuda.empty_cache()
