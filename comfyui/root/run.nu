#!/usr/bin/env nu

# ==============================
# Environment variables
# ==============================

# Enable HIP backend for AMD GPUs
#$env.COMFYUI_USE_HIP = "1"
#$env.SAM3_FORCE_CPU = "1"
# MIOpen / ROCm tuning
#$env.MIOPEN_FIND_MODE = "2"                 # 2 = aggressive find, caches best kernels
#$env.MIOPEN_USE_ROCBLAS = "1"               # Prefer rocBLAS for GEMMs
#$env.MIOPEN_GEMM_ENFORCE_BACKEND = "1"       # Force backend choice
#$env.MIOPEN_COMPILE_PARALLEL_LEVEL = "16"    # Parallel kernel compilation
#$env.USE_FLASH_ATTN = true                        # Flash attention is unstable on ROCm, disable it
#$env.OLD_GPU = true
#$env.COMFYUI_SAM3_FORCE_CPU_OFFLOAD = "1"  # Force SAM3 model offloading to CPU to save VRAM
#$env.TORCH_USE_HIP_DSA = "1"  # Allow SAM3 to use GPU if available
#$env.AMD_SERIALIZE_KERNEL = "3"
#$env.AMD_ENABLE_MIOPEN_ENV = "0"  # Enable MIOpen environment variable support
# Backstop if anything re-enables MIOpen (--disable-miopen below turns it off): heuristic kernel
# pick, no per-shape search.
$env.MIOPEN_FIND_MODE = "FAST"
$env.SPEED_DEBUG = "0"
$env.SPEED_DEBUG_DIR = "/media/luna/work/cache/speed-debug"


let PROFILE_MODE = "0";

if PROFILE_MODE == "1" {
  $env.COVERAGE_PROCESS_START = ".coveragerc"
  $env.COMFY_PROFILE = "1"
}

# ==============================
# Disable core dumps (ulimit -c 0)
# ==============================
ulimit -c 0

# ==============================
# Whitelisted custom nodes
# ==============================
# Only these custom_nodes folders will be loaded,
# even though --disable-all-custom-nodes is enabled.
let whitelist_custom_nodes = [
  #SwarmComfyExtra      # SwarmUI integration helpers
  #SwarmComfyCommon     # Shared SwarmUI node utilities
  ComfyUI-GGUF         # GGUF / LLM loader nodes
  ComfyUI-Gfx906Int4   # gfx906 int4 quantization test node
  ComfyUI-ResolutionGrowth # compose at native size, grow to the target in small tiled steps
  ComfyUI-Flux2Klein-Enhancer
  #AspectRatioResolution # Custom aspect ratio node with better behavior
  #MaterialSaver	  # Advanced prompt composition node
  #ComfyUI-DepthAnythingV3
  #ComfyUI-SAM3
  ComfyUI-MultiGPU
  #ComfyUI-Florence2
  #ComfyUI-RMBG
  ComfyUI_TiledKSampler
  #ComfyUI_Luna_QOL_Nodes
  ComfyUI-PiD
  ComfyUI-ForLoops
  #ImageBatchPostprocess
  #ComfyUI-SPEED
  ComfyUI-ConditioningKrea2Rebalance
  ComfyUI-Krea2-Projector-Tuner
  comfyui-krea2-controlnet
  ComfyUI-Krea2TextEncoder
]

# ==============================
# Main argument list
# ==============================
let args = [
  # Enable specific fast-path optimizations
  # Valid values include: fp16_accumulation, fp8_matrix_mult, cublas_ops
  "--fast=fp16_accumulation"
  #"--disable-dynamic-vram"
  # Precision control
  # VAE in FP16. Depends on --disable-miopen below: with MIOpen, fp16 was ~1.5x SLOWER than fp32
  # and once gave a nondeterministic 0.52 pixel error (vae_bench.py). Without it (rocBLAS convs),
  # fp16 is 1.43x faster (2560x1440 tiled decode 2.42 vs 3.47 s, encode 1.43 vs 2.02 s), 65-82 dB
  # vs fp32 on real renders, finite on stress images (develop/experiments/2026-09-29/
  # vae_int8_bench.py, vae_fp16_check.py). If MIOpen comes back on, go back to --fp32-vae.
  "--fp16-vae"
  #"--cpu-vae"
  #"--fp16-unet"        # Run UNet in FP16
  "--force-fp16"       # Force FP16 everywhere possible
  "--gpu-only"
  #"--fp16-text-enc"    # Text encoders in FP16
  #"--enable-triton-backend"
  #"--disable-nvml-pressure"
  #"--disable-smart-memory"
  #"--lowvram"        # Use normal VRAM allocation strategy
  # Device selection
  #"--default-device=1"
  "--force-non-blocking"
  #"--cuda-malloc"
  #"--disable-cuda-malloc"
  #"--use-flash-attention"
  #"--use-sage-attention"
  #"--use-split-cross-attention"
  "--use-quad-cross-attention"
  #"--use-pytorch-cross-attention"
  "--fp16-intermediates" # Keep intermediate tensors in FP16 to save VRAM
  # Live sampler previews: tiny TAE decoder from models/vae_approx (Krea2 = lighttaew2_1);
  # falls back to the linear latent2rgb preview if the file is missing.
  # Without this flag the server default is no previews.
  "--preview-method=taesd"
  # No MIOpen: each new conv size cost ~8 s of kernel building (then 0.39 s per VAE decode at
  # ~1400x850), never pays off for changing sizes, and has been flaky here. Without it: no
  # per-size cost, ~1.2 s per decode (develop/experiments/2026-09-29/vae_shape_test.py).
  "--disable-miopen"
  # Memory behavior
  #"--async-offload=8"  # Async weight offloading with 8 streams
                       # Helps hide PCIe latency on large models
  #"--disable-async-offload"

  # Disable features that are unstable or unnecessary on ROCm
  "--disable-xformers" # xformers is CUDA-centric and flaky on AMD

  # Practicality over subscriptions
  "--disable-api-nodes"        # Not paying for flaky third-party API providers

  # Startup speed > feature hoarding
  "--disable-all-custom-nodes" # Suppress ComfyUI-Manager and its slow startup scans

  # Load only the custom nodes that are actually needed
  "--whitelist-custom-nodes"
  ...$whitelist_custom_nodes

  # VRAM strategy
  #"--lowvram"          # Aggressive VRAM conservation
  "--listen"
  "--port=11900"
  "--enable-cors-header"
]

# ==============================
# Debug: show final argv
# ==============================
print $"Running: python3 main.py ($args | str join ' ')"

# ==============================
# Execute
# ==============================

if PROFILE_MODE == "1" {
  coverage run main.py ...$args
} else {
  python3 main.py ...$args
}
