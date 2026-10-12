"""NTK-by-parts (YaRN) RoPE scaling against the 2560x1440 artifacts, after
rope_test.py showed the trade-off: plain PI kills mirroring/splotches but
fragments fine detail (it squeezes the high-frequency neighbour bands too);
DyPE keeps detail but lets some splotchiness back.
YaRN rescales each spatial RoPE frequency by how many rotations it makes over
the training span: >= BETA rotations stay exact (local detail), <= ALPHA get
full interpolation by s (global layout), linear ramp between. Optional
attention temperature (YaRN's 0.1 ln(1/s) + 1, applied to q and k).
Also a DyPE exponent sweep (later release of the squeeze).

Usage: yarn_test.py <variant> [...]  (see VARIANTS; "all")
Saves develop/out/progressive/yarn_<variant>.png; prints time + detail."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import progressive as prog  # noqa: E402  (bootstrap, loaders, helpers; its main loop is guarded)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from comfy.ldm.flux.layers import EmbedND  # noqa: E402
from comfy.ldm.flux.math import rope  # noqa: E402
import pipeline  # noqa: E402

NATIVE_TOKENS = 64 * 64  # 1024x1024 / (8 * 2)^2
TRAIN_SPAN = 64          # tokens per side at the ~1 MP training size
TARGET = (2560, 1440)
# name: ("yarn", alpha, beta, temperature) | ("dype", power, floor)
#     | ("yarn_dype", alpha, beta, power): YaRN band split, the squeeze s released over
#       sigma as s ** (sigma ** power) (Luna: dype_p4 fewest splotches, yarn_a0.5_b4 least
#       fragmentation)
VARIANTS = {
    "yarn_dype_a0.5_b4_p4": ("yarn_dype", 0.5, 4.0, 4.0),
    "yarn_dype_a1_b8_p4": ("yarn_dype", 1.0, 8.0, 4.0),
    "yarn_a1_b8": ("yarn", 1.0, 8.0, False),
    "yarn_a1_b8_temp": ("yarn", 1.0, 8.0, True),
    "yarn_a0.5_b4": ("yarn", 0.5, 4.0, False),
    "dype_p4": ("dype", 4.0, 1.0),
}


def interpolation_scale(ids):
    tokens = ids.shape[1]
    return min(1.0, math.sqrt(NATIVE_TOKENS / tokens))


class YarnEmbedND(EmbedND):
    """EmbedND with per-frequency interpolation on the spatial axes (1, 2)."""
    def __init__(self, base, alpha, beta, temperature, dype_power=None):
        super().__init__(base.dim, base.theta, base.axes_dim)
        self.alpha, self.beta, self.temperature = alpha, beta, temperature
        self.dype_power = dype_power
        self.sigma = 1.0  # set per call by sigma_recorder when dype_power is used

    def spatial_rope(self, pos, dim, s):
        # Same frequencies as flux rope(); pos is rescaled per frequency instead of omega,
        # so each band k sees pos * f_k with f_k in [s, 1].
        k = torch.arange(dim // 2, dtype=torch.float64)
        omega = 1.0 / (self.theta ** (2 * k / dim))
        rotations = TRAIN_SPAN * omega / (2 * math.pi)
        gamma = ((rotations - self.alpha) / (self.beta - self.alpha)).clamp(0, 1)
        factor = gamma + (1 - gamma) * s
        out = torch.einsum("...n,d->...nd", pos.to(torch.float64).cpu(), omega * factor)
        out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
        return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32, device=pos.device)

    def forward(self, ids):
        s = interpolation_scale(ids)
        if self.dype_power is not None:
            s = s ** (self.sigma ** self.dype_power)
        parts = [rope(ids[..., 0], self.axes_dim[0], self.theta)]
        parts += [self.spatial_rope(ids[..., i], self.axes_dim[i], s) for i in (1, 2)]
        emb = torch.cat(parts, dim=-3)
        if self.temperature and s < 1.0:
            # q and k are both rotated by emb, so logits scale by m^2 = YaRN's 1/t
            emb = emb * math.sqrt(0.1 * math.log(1 / s) + 1)
        return emb.unsqueeze(1)


def sigma_recorder(embedder):
    """post_input patch: the embedder sees only ids, so hand it the current sigma."""
    def patch(args):
        embedder.sigma = float(args["transformer_options"]["sigmas"].flatten()[0])
        return args
    return patch


def dype_patch(power, floor):
    def patch(args):
        ids = args["img_ids"]
        sigma = float(args["transformer_options"]["sigmas"].flatten()[0])
        s = interpolation_scale(ids) ** (sigma ** power)
        s = min(s, floor)
        if s < 1.0:
            ids = ids.clone()
            ids[..., 1:] *= s
        return {**args, "img_ids": ids}
    return patch


if __name__ == "__main__":
    names = list(VARIANTS) if sys.argv[1:] == ["all"] else sys.argv[1:]
    resolved = pipeline.resolve({"kind": "moodykrea_int4", "prompt": prog.PROMPT,
                                 "latent": {"width": prog.NATIVE[0], "height": prog.NATIVE[1]}})
    with torch.inference_mode():
        base_model, clip, vae = pipeline.build_loaders(resolved)
        (fused,) = pipeline.call_node("Gfx906FusedAttention", model=base_model, qk="int4", pv="u8", skip_below=1e-4,
                                      sparse_tau=0.0, max_missed=0.02)
        (positive,) = pipeline.call_node("CLIPTextEncode", clip=clip, text=prog.PROMPT)
        base_embedder = fused.model.diffusion_model.pe_embedder
        for name in names:
            spec = VARIANTS[name]
            model = fused.clone()
            if spec[0] == "yarn":
                model.add_object_patch("diffusion_model.pe_embedder", YarnEmbedND(base_embedder, *spec[1:]))
            elif spec[0] == "yarn_dype":
                alpha, beta, power = spec[1:]
                embedder = YarnEmbedND(base_embedder, alpha, beta, False, dype_power=power)
                model.add_object_patch("diffusion_model.pe_embedder", embedder)
                model.set_model_patch(sigma_recorder(embedder), "post_input")
            else:
                model.set_model_patch(dype_patch(*spec[1:]), "post_input")
            t0 = time.perf_counter()
            (latent,) = pipeline.call_node("EmptyLatentImage", width=TARGET[0], height=TARGET[1], batch_size=1)
            (latent,) = pipeline.call_node("KSamplerAdvanced", model=model, add_noise="enable", noise_seed=prog.SEED,
                                           steps=8, cfg=1.0, sampler_name="er_sde", scheduler="simple",
                                           positive=positive, negative=positive, latent_image=latent, start_at_step=0,
                                           end_at_step=8, return_with_leftover_noise="disable")
            (images,) = pipeline.call_node("VAEDecodeTiled", samples=latent, vae=vae, tile_size=512, overlap=64,
                                           temporal_size=64, temporal_overlap=8)
            arr = (images[0].float().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(prog.OUT, f"yarn_{name}.png"))
            print(json.dumps({"variant": name, "total_s": round(time.perf_counter() - t0, 1),
                              "detail": round(prog.detail(arr), 2)}), flush=True)
