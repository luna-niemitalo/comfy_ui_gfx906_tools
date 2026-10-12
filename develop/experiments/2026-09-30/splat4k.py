"""Continuous splat queue at a 4K target (Luna): no passes -- every model call takes the
SPLATS neediest 4x4-token splats (box write-back, 1-token clean context ring, true positions;
splat_test.py C: best polish), until stopped. Per token: spp (samples so far) and change (EMA
of the RMS move of its last samples, starts high so everything is visited once first).
Splat priority = (max change in footprint + FLOOR) / (1 + min spp in footprint); the call's
sigma = sigma(mean spp of its splats), sigma(spp) = max(SIGMA_FLOOR, sigma_start * DECAY**spp).
Low band anchored to the composition with weight ANCHOR * (sigma / 0.5)**2 (inside the splats).
Prompt: the full one -- every call spans the whole image (tiles take the detail prompt because
they see only a region).

Runs (2 GPUs: growth via the row-split node, 4 tiles per call; the queue runs two independent
splat calls per iteration, one on each GPU -- the split node sends calls with reference latents to
one GPU, so the second call uses its replica directly):
  reference  linear growth 1280x720 -> 3840x2160 (the node's Grower, step 256 px), time T
  A_<rope>   the reference's own state after its last step with sigma >= A_SIGMA, Lanczos to
             4K, queue from A_SIGMA; the clock includes the growth prefix
  B_<rope>   composition, Lanczos to 4K, queue from B_SIGMA
  rope: stock (true positions, up to 240x135 tokens) | yarn (a1_b8, fixed s = 64/sqrt(tokens))
Snapshots at SNAPS x T: out/splat4k/<run>_t<frac>.png + one JSON line each (sigma, spp,
change, Immerkaer noise at full res, detail at 1280x720).

Usage: splat4k.py [A_stock A_yarn B_stock B_yarn]   (default A_yarn B_yarn; the reference always runs)"""
import asyncio, json, math, os, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "2026-09-29"))
import splat_test as st  # noqa: E402  (bootstrap via progressive; Predictor, strip helpers)
import progressive as prog  # noqa: E402
import tiled_test as tt  # noqa: E402
import yarn_test as yt  # noqa: E402

import nodes  # noqa: E402
NODE_DIR = os.path.join(os.path.dirname(nodes.__file__), "custom_nodes", "ComfyUI-ResolutionGrowth")
asyncio.run(nodes.load_custom_node(NODE_DIR))  # progressive.py's whitelist excludes it

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from comfy.ldm.flux.math import rope  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import pipeline  # noqa: E402

growth = sys.modules[next(name for name in sys.modules if name.endswith("ComfyUI-ResolutionGrowth.growth"))]
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/splat4k"
os.makedirs(OUT, exist_ok=True)
TARGET = (3840, 2160)
STEP_PX = 256
K = 5                   # 4 -> 5 (Luna): ring share 1.25 -> 0.96 context tokens per active token
SPLATS = 164            # per call, one call per GPU: ~4100 active tokens (+ ~4k context) each
CHANGE_INIT, FLOOR, EMA = 1.0, 0.02, 0.5
DECAY, SIGMA_FLOOR = 0.7, 0.05
A_SIGMA, B_SIGMA = 0.2, 0.5
ANCHOR = 1.0
SNAPS = (0.25, 0.5, 0.75, 1.0, 1.25)
# modes (against the grain of the first 4K run):
#   replace  newest prediction wins inside the splat (box)
#   settle   replace until 1.0 x budget, then one full-frame tiled pass (growth's sampler) at SETTLE_SIGMA
#   accum    below ACC_SIGMA a weighted running mean per latent pixel (flat-top kernel, rim 0.25, so
#            overlapping splats of different calls average at their rims -- Luna's "splats as pixels");
#            the state before counts as PRIOR samples; a replace over accumulated pixels restarts them
SETTLE_SIGMA = 0.05
ACC_SIGMA, PRIOR = 0.12, 1.0
RUNS = sys.argv[1:] or ["A_yarn_settle", "A_yarn_accum"]  # Luna picked YaRN for 4K positions (2026-10-01)


class FixedYarn(yt.YarnEmbedND):
    """yarn_test's YaRN embedder with the squeeze s fixed (its own s counts the ids it gets:
    here active + context tokens, not the image)."""

    def __init__(self, base, s):
        super().__init__(base, 1.0, 8.0, False)
        self.s = s

    def forward(self, ids):
        parts = [rope(ids[..., 0], self.axes_dim[0], self.theta)]
        parts += [self.spatial_rope(ids[..., i], self.axes_dim[i], self.s) for i in (1, 2)]
        return torch.cat(parts, dim=-3).unsqueeze(1)


def to_np(image):
    return (image[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)


def measure(arr):
    small = np.asarray(Image.fromarray(arr).resize(prog.NATIVE, Image.LANCZOS))
    return {"noise": round(growth.noise_sigma(torch.from_numpy(arr)[None].float() / 255), 3),
            "detail_720": round(prog.detail(small), 2)}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reference(model, vae, detail_positive, positive, composition):
    """Linear growth; returns (final latent, T, (prefix latent, prefix seconds) for path A)."""
    grid = growth.snap_px(vae)
    sizes = growth.linear_sizes(prog.NATIVE, TARGET, STEP_PX, grid)
    schedule = growth.make_schedule(model, prog.NATIVE, sizes, 0.5, 0.0, 1.0, 2.0)
    grower = growth.Grower(model, detail_positive, positive, vae, composition, 1.0, prog.SEED + 1,
                           1024, 256, 4, "lanczos", None)  # 4 tiles/call = 2 per GPU (batch split), as scale_test
    x = composition
    prefix = None
    t0 = time.perf_counter()
    for k, (size, sigma, anchor) in enumerate(zip(schedule["sizes"], schedule["sigmas"], schedule["anchors"])):
        if sigma < A_SIGMA and prefix is None:
            sync()
            prefix = (x, time.perf_counter() - t0, sigma)
        x = grower.step(k, x, size, sigma, anchor)
        sync()
        print(json.dumps({"run": "reference", "step": k + 1, "size": f"{size[0]}x{size[1]}", "sigma": round(sigma, 3),
                          "t": round(time.perf_counter() - t0, 1), **{n: round(v, 2) for n, v in grower.timing.items()}}),
              flush=True)
    total = time.perf_counter() - t0
    arr = to_np(growth.decode(vae, x))
    Image.fromarray(arr).save(os.path.join(OUT, "reference.png"))
    print(json.dumps({"run": "reference", "total_s": round(total, 1), "prefix_s": round(prefix[1], 1),
                      "prefix_next_sigma": round(prefix[2], 3), **measure(arr)}), flush=True)
    return total, prefix


def neediest(spp_tok, change, rng):
    """Up to SPLATS non-overlapping (1-token gap) splat origins, highest priority first."""
    chg = F.max_pool2d(change[None, None], K, stride=1)[0, 0]
    spp_min = -F.max_pool2d(-spp_tok[None, None], K, stride=1)[0, 0]
    score = ((chg + FLOOR) / (1 + spp_min)).cpu().numpy()
    return st.place_splats(score, K, 1, 2 * SPLATS * K * K, rng, 0.01 * float(score.max()))


def pair(pred, x_t, sigma, groups, rings, x0, replica):
    """groups[0] on the main GPU, groups[1] on the replica, in parallel; predictions concatenated."""
    peer = {}

    def run_peer():
        try:
            with torch.cuda.device(replica[1]), torch.inference_mode():
                peer["out"] = pred(x_t, sigma, groups[1], rings[1], x0, replica=replica)
                torch.cuda.synchronize()
        except BaseException as e:
            peer["error"] = e

    thread = threading.Thread(target=run_peer, name="splat-peer")
    thread.start()
    try:
        out0 = pred(x_t, sigma, groups[0], rings[0], x0)
    finally:
        thread.join()
    if "error" in peer:
        raise peer["error"]
    return torch.cat((out0, peer["out"]), dim=3)


def split_replica(model):
    """The row-split wrapper's model copy on the second GPU (made during the growth reference)."""
    wrappers = model.wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL]
    split = next(w for ws in wrappers.values() for w in ws if hasattr(w, "replica"))
    if split.replica is None:
        raise RuntimeError(f"expected the row split's replica on the second GPU after growth, got none "
                           f"(peer {split.device}: fell back to one GPU?)")
    return split.replica, split.device


def settle(model, detail_positive, positive, x0, pred):
    """One full-frame step at SETTLE_SIGMA through growth's tiled sampler (tile-local positions,
    detail prompt, both GPUs via the row split) -- what growth's last step does. Model space in/out."""
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(detail_positive, positive)
    guider.set_cfg(1.0)
    latent = model.model.process_latent_out(x0)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(prog.SEED + 50))
    sampler = growth.tiled_x0_sampler(None, 0.0, 128, 32, 4, {})
    out = guider.sample(noise, latent, sampler, torch.tensor([SETTLE_SIGMA, 0.0]), seed=prog.SEED + 50,
                        disable_pbar=True)
    return model.model.process_latent_in(out.to(pred.device)).float()


def queue(label, pred, model, vae, start, ref, sigma_start, clock0, budget, replica, mode, spp0=None, out=None):
    """Runs until the last snapshot (settle: 1.0 x budget); returns the model-space latent.
    spp0: [h, w] starting samples per token (oc_splat.py: grown parts start 'already refined');
    out: snapshot directory (default OUT)."""
    out = out or OUT
    """Run the splat queue from `start` (model space, 4K) until clock0 + max(SNAPS) * budget."""
    h, w = pred.h, pred.w
    x0 = start.clone()
    spp_tok = torch.zeros(h, w, device=pred.device) if spp0 is None else spp0.to(pred.device).clone()
    change = torch.full((h, w), CHANGE_INIT, device=pred.device)
    rng = np.random.default_rng(3)
    g = torch.Generator().manual_seed(prog.SEED + 40)
    acc = torch.zeros_like(x0)
    wsum = torch.zeros(x0.shape[-2:], device=pred.device)
    kernel = st.kernel(K, "flat25", pred.device)
    calls = 0
    paused = 0.0  # snapshot decodes are not charged
    snaps = [s for s in SNAPS if mode != "settle" or s <= 1.0]
    t_start = time.perf_counter()
    while snaps:
        elapsed = clock0 + time.perf_counter() - paused - t_start
        if elapsed >= snaps[0] * budget:
            t_snap = time.perf_counter()
            arr = to_np(growth.decode(vae, model.model.process_latent_out(x0)))
            Image.fromarray(arr).save(os.path.join(out, f"{label}_t{snaps[0]}.png"))
            print(json.dumps({"run": label, "t_frac": snaps[0], "t": round(elapsed, 1), "calls": calls,
                              "spp_min": round(float(spp_tok.min()), 2), "spp_mean": round(float(spp_tok.mean()), 2),
                              "change_max": round(float(change.max()), 4),
                              "change_mean": round(float(change.mean()), 4), **measure(arr)}), flush=True)
            snaps.pop(0)
            paused += time.perf_counter() - t_snap
            continue
        origins = neediest(spp_tok, change, rng)
        # two independent calls, one per GPU, interleaved in priority order so both spread
        groups = [torch.tensor(st.splat_tokens(origins[r::2], K, w), device=pred.device) for r in (0, 1)]
        idx = torch.cat(groups)
        sigma = max(SIGMA_FLOOR, sigma_start * DECAY ** float(spp_tok.flatten()[idx].mean()))
        eps = torch.randn(x0.shape, generator=g).to(pred.device)
        x_t = (1 - sigma) * x0 + sigma * eps
        rings = [st.ring(group, h, w, pred.device, 1) for group in groups]
        tok = st.tokens_of(x0).clone()
        old = tok[:, :, :, idx].clone()
        new = pair(pred, x_t, sigma, groups, rings, x0, replica)
        tok[:, :, :, idx] = new
        target = st.untokens(tok, h, w)
        anchor = ANCHOR * (sigma / 0.5) ** 2
        low = growth.resize(target, ref.shape[-2:], "area")
        target = target + anchor * growth.resize(ref - low, target.shape[-2:], "bicubic")
        footprint = torch.zeros(h * w, device=pred.device)
        footprint[idx] = 1
        mask = F.interpolate(footprint.view(1, 1, h, w), scale_factor=2, mode="nearest")[0, 0]
        if mode == "accum" and sigma < ACC_SIGMA:
            weight = torch.zeros_like(wsum)
            for r, c in origins:
                weight[2 * r:2 * r + 2 * K, 2 * c:2 * c + 2 * K] = kernel
            fresh = (weight > 0) & (wsum == 0)
            acc = torch.where(fresh, PRIOR * x0, acc)
            wsum = torch.where(fresh, torch.full_like(wsum, PRIOR), wsum)
            acc = acc + weight * target
            wsum = wsum + weight
            x0 = torch.where(wsum > 0, acc / wsum.clamp(min=1e-8), x0)
        else:
            x0 = x0 + mask * (target - x0)
            wsum = wsum * (1 - mask)
        moved = (new - old).pow(2).mean(dim=(0, 1, 2, 4, 5)).sqrt()
        flat_change = change.view(-1)
        flat_change[idx] = EMA * flat_change[idx] + (1 - EMA) * moved
        spp_tok.view(-1)[idx] += 1
        sync()  # ROCm 7.2 AsyncEventsLoop crash with deep unsynced queues (growth.py)
        calls += 1
    return x0


if __name__ == "__main__":
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (model,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (model,) = pipeline.call_node("Gfx906RowSplitKrea2", model=model, device="auto", min_image_tokens=2048)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        (detail_positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=tt.DETAIL_PROMPT)
        (latent,) = pipeline.call_node("EmptyLatentImage", width=prog.NATIVE[0], height=prog.NATIVE[1], batch_size=1)
        (comp,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                     steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple", positive=positive,
                                     negative=positive, latent_image=latent, start_at_step=0, end_at_step=8,
                                     return_with_leftover_noise="disable")
        composition = comp["samples"]
        # the composition is deterministic, so a finished reference is reused (delete the file to rerun it)
        cached = os.path.join(OUT, "reference_state.pt")
        if os.path.exists(cached):
            state = torch.load(cached)
            budget, prefix, prefix_s = state["budget"], state["prefix"], state["prefix_s"]
            print(json.dumps({"run": "reference", "cached": cached, "total_s": round(budget, 1)}), flush=True)
        else:
            budget, (prefix, prefix_s, _) = reference(model, vae, detail_positive, positive, composition)
            torch.save({"budget": budget, "prefix": prefix.cpu(), "prefix_s": prefix_s}, cached)

        shape_4k = (1, 1, 1, TARGET[1] // 8, TARGET[0] // 8)
        pred = st.Predictor(model, positive, shape_4k)
        ref = model.model.process_latent_in(composition.to(pred.device)).float()
        replica = split_replica(model)
        embedder = model.model.diffusion_model.pe_embedder
        yarn = FixedYarn(embedder, 64 / math.sqrt(pred.h * pred.w))
        starts = {}
        for label in RUNS:
            path, rope_kind, *rest = label.split("_")
            mode = rest[0] if rest else "replace"
            if path not in starts:
                t0 = time.perf_counter()
                src = prefix if path == "A" else composition
                up = growth.pixel_upscale(vae, src, TARGET, "lanczos", None)
                sync()
                up_s = time.perf_counter() - t0
                starts[path] = (model.model.process_latent_in(up.to(pred.device)).float(), up_s)
                Image.fromarray(to_np(growth.decode(vae, up))).save(os.path.join(OUT, f"{path}_start.png"))
            start, up_s = starts[path]
            clock0 = (prefix_s if path == "A" else 0.0) + up_s
            # the replica is a deepcopy with its own embedder: both GPUs get the same one
            model.model.diffusion_model.pe_embedder = yarn if rope_kind == "yarn" else embedder
            replica[0].pe_embedder = yarn if rope_kind == "yarn" else embedder
            try:
                x0 = queue(label, pred, model, vae, start, ref, A_SIGMA if path == "A" else B_SIGMA, clock0, budget,
                           replica, mode)
            finally:
                model.model.diffusion_model.pe_embedder = embedder
                replica[0].pe_embedder = embedder
            if mode == "settle":
                t0 = time.perf_counter()
                x0 = settle(model, detail_positive, positive, x0, pred)
                sync()
                settle_s = time.perf_counter() - t0
                arr = to_np(growth.decode(vae, model.model.process_latent_out(x0)))
                Image.fromarray(arr).save(os.path.join(OUT, f"{label}_settled.png"))
                print(json.dumps({"run": label, "settled": True, "settle_s": round(settle_s, 1),
                                  "t": round(budget + settle_s, 1), **measure(arr)}), flush=True)
        print(json.dumps({"done": True}), flush=True)
