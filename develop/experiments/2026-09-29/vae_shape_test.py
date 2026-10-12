"""First-use cost of a new VAE decode size (Luna: MIOpen's per-shape work must go for
smooth resolution growth, where sizes never repeat). One fresh process per MIOpen setting,
run with env vars set by the caller:
  default   : as run.nu (no MIOPEN_* set)
  fast      : MIOPEN_FIND_MODE=FAST (heuristic pick, no benchmarking)
  gemm      : direct/winograd/implicit-GEMM/FFT solvers off -> im2col + rocBLAS GEMM only
              (precompiled, nothing built per shape)
  nomiopen  : torch.backends.cudnn.enabled = False -- MIOpen not used at all; PyTorch's own
              im2col + rocBLAS convolution (Luna: disable MIOpen completely if possible)
Each run decodes 3 never-seen sizes (distinct per mode, since the kernel cache is shared
on disk) twice: first call = per-shape cost, second = steady state.

Usage: vae_shape_test.py <mode> [device]"""
import os, sys, time
sys.path.insert(0, "/media/luna/work/tinker/ComfyUI/develop/harness")
import bootstrap  # noqa: E402,F401
import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

MODES = {"default": 0, "fast": 1, "gemm": 2, "nomiopen": 3}
mode = sys.argv[1]
torch.backends.cudnn.enabled = mode != "nomiopen"
device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:1")
torch.cuda.set_device(device)
print(f"mode {mode}: MIOPEN env {{{', '.join(f'{k}={v}' for k, v in os.environ.items() if k.startswith('MIOPEN'))}}}",
      flush=True)
sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", "qwenImageVAE_qwenImageVAE.safetensors"))
with torch.inference_mode():
    vae = comfy.sd.VAE(sd=sd, device=device, dtype=torch.float32)
    # latent sizes (h, w) not used by any earlier run; offset per mode
    for i in range(3):
        h, w = 94 + 2 * (3 * MODES[mode] + i), 166 + 2 * (3 * MODES[mode] + i)
        latent = torch.randn(1, 16, 1, h, w, generator=torch.Generator().manual_seed(i))
        times = []
        for _ in range(2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            vae.decode_tiled(latent, tile_x=64, tile_y=64, overlap=8)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"{mode} {w * 8}x{h * 8}: first {times[0]:.2f} s, second {times[1]:.2f} s", flush=True)
