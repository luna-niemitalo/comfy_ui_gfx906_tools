"""Fused gfx906 attention: correctness vs fp32 reference, speed vs comfy quad attention."""
import glob, importlib, json, os, sys, time
REPO = "/media/luna/work/tinker/ComfyUI"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "custom_nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.use_quad_cross_attention = True
import torch  # noqa: E402
att = importlib.import_module("ComfyUI-Gfx906Int4.attention")
from comfy.ldm.modules.attention import attention_sub_quad  # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
torch.cuda.set_device(dev)
CAP = "/media/luna/work/cache/tmp/claude-1000/-media-luna-work-tinker-ComfyUI/a7cb27a3-6733-41dc-80cc-6bbddc7af53b/scratchpad/attn_capture"
MODES = [(qk, pv) for qk in ("fp16", "int8", "int4") for pv in ("fp16", "u8")]


def reference(q, k, v):
    rep = q.shape[1] // k.shape[1]
    kf, vf = k.float().repeat_interleave(rep, 1), v.float().repeat_interleave(rep, 1)
    out = torch.empty(q.shape, device=q.device)
    for h0 in range(0, q.shape[1], 8):
        s = q[:, h0:h0 + 8].float() @ kf[:, h0:h0 + 8].transpose(-1, -2) * q.shape[-1] ** -0.5
        out[:, h0:h0 + 8] = torch.softmax(s, -1) @ vf[:, h0:h0 + 8]
    return out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], -1)


def rel(a, b):
    return float(f"{((a.float() - b).norm() / b.norm()).item():.2e}")


def check(name, q, k, v):
    ref = reference(q, k, v)
    rec = {"case": name, "L": q.shape[2]}
    for qk, pv in MODES:
        out = att.fused_attention(q, k, v, qk, pv)
        rec[f"{qk}/{pv}"] = rel(out, ref) if torch.isfinite(out).all() else "nonfinite"
    print(json.dumps(rec), flush=True)


torch.manual_seed(0)
q = torch.randn(1, 8, 300, 128, device=dev, dtype=torch.float16) * 2
k = torch.randn(1, 2, 300, 128, device=dev, dtype=torch.float16) * 2
v = torch.randn(1, 2, 300, 128, device=dev, dtype=torch.float16)
check("random_L300", q, k, v)
for path in sorted(glob.glob(os.path.join(CAP, "*.pt")))[::3]:
    d = torch.load(path)
    check(os.path.basename(path)[:-3], d["q"].to(dev), d["k"].to(dev), d["v"].to(dev))


def time_ms(fn, iters=3):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


for L in (2444, 14540):
    q = torch.randn(1, 48, L, 128, device=dev, dtype=torch.float16)
    k = torch.randn(1, 12, L, 128, device=dev, dtype=torch.float16)
    v = torch.randn(1, 12, L, 128, device=dev, dtype=torch.float16)
    ke, ve = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
    rec = {"L": L, "quad_ms": round(time_ms(lambda: attention_sub_quad(q, ke, ve, 48, skip_reshape=True)), 1)}
    for qk, pv in MODES:
        rec[f"{qk}/{pv}_ms"] = round(time_ms(lambda: att.fused_attention(q, k, v, qk, pv)), 1)
    print(json.dumps(rec), flush=True)
    del q, k, v, ke, ve
