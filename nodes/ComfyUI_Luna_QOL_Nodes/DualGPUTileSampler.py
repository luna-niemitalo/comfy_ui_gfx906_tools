import threading
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import nodes

class DualGPUTileSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_0":      ("MODEL",),
            "model_1":      ("MODEL",),
            "positive":     ("CONDITIONING",),
            "negative":     ("CONDITIONING",),
            "latent_tiles": ("LATENT",),   # batch dim = N tiles
            "seed":         ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps":        ("INT",    {"default": 20, "min": 1}),
            "cfg":          ("FLOAT",  {"default": 7.0}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
            "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
            "denoise":      ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/multigpu"

    def _sample_tile(self, model, tile_latent, positive, negative,
                     seed, steps, cfg, sampler_name, scheduler, denoise):
        noise = comfy.sample.prepare_noise(tile_latent, seed, None)

        return comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, tile_latent,
            denoise=denoise,
            disable_noise=False,
            disable_pbar=True,   # critical: pbar is not thread-safe
        )

    def sample(self, model_0, model_1, positive, negative, latent_tiles,
               seed, steps, cfg, sampler_name, scheduler, denoise):
        print(f"Sampling {latent_tiles['samples'].shape[0]} tiles with dual GPU")
        tiles = latent_tiles["samples"]   # [N, C, H, W]
        N = tiles.shape[0]
        print(f"Sampling {N} tiles with dual GPU")
        models = [model_0, model_1]
        results = [None] * N

        # --- Phase 1: sequential, proves dual-model works ---
        for i in range(N):
            m = models[i % 2]
            tile = tiles[i:i+1]           # keep batch dim
            results[i] = self._sample_tile(
                m, tile, positive, negative,
                seed + i, steps, cfg, sampler_name, scheduler, denoise
            )
            print(f"[DualGPU] tile {i}/{N} done on {m.load_device}")

        # --- Phase 2: replace the loop above with this once Phase 1 passes ---
        # errors = [None] * N
        # def worker(i):
        #     try:
        #         m = models[i % 2]
        #         tile = tiles[i:i+1].clone()   # clone: each thread needs its own tensor
        #         results[i] = self._sample_tile(
        #             m, tile, positive, negative,
        #             seed + i, steps, cfg, sampler_name, scheduler, denoise
        #         )
        #     except Exception as e:
        #         errors[i] = e
        #
        # threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        # for t in threads: t.start()
        # for t in threads: t.join()
        # for i, e in enumerate(errors):
        #     if e: raise RuntimeError(f"Tile {i} failed: {e}")

        return ({"samples": torch.cat(results, dim=0)},)

