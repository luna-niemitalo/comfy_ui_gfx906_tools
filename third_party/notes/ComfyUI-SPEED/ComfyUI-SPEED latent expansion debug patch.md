```diff
diff --git a/speed_sampler.py b/speed_sampler.py
index 0000000..0000000 100644
--- a/speed_sampler.py
+++ b/speed_sampler.py
@@ -1,6 +1,8 @@
 """ComfyUI sampler for Spectral Progressive Diffusion.
 Based on https://github.com/howardhx/speed
 Wraps any ``comfy.k_diffusion.sampling.sample_*`` solver with the
 resolution transitions from the paper.
 """
 from __future__ import annotations

+import json
+import os
+from pathlib import Path
 from typing import List, Tuple

 import numpy as np
 import torch

@@ -14,6 +16,67 @@ from .spectral_utils import (
     validate_scales,
 )

+
+# =============================================================================
+# Debug latent dumping
+# =============================================================================
+
+def _debug_enabled() -> bool:
+    """Return whether SPEED latent debugging is enabled.
+
+    IN:
+        SPEED_DEBUG=1/true/yes/on enables dumping.
+    OUT:
+        Boolean indicating whether debug output should be generated.
+    """
+    return os.environ.get("SPEED_DEBUG", "").lower() in {
+        "1", "true", "yes", "on"
+    }
+
+
+def _debug_dir() -> Path:
+    """Return the directory used for SPEED debug artifacts."""
+    return Path(
+        os.environ.get(
+            "SPEED_DEBUG_DIR",
+            "/tmp/comfyui-speed-debug",
+        )
+    )
+
+
+def _latent_stats(x: torch.Tensor) -> dict:
+    """Return cheap diagnostic statistics for a latent."""
+    y = x.detach().float()
+    return {
+        "shape": list(x.shape),
+        "dtype": str(x.dtype),
+        "device": str(x.device),
+        "numel": x.numel(),
+        "min": float(y.min().item()),
+        "max": float(y.max().item()),
+        "mean": float(y.mean().item()),
+        "std": float(y.std().item()),
+        "rms": float(torch.sqrt(torch.mean(y * y)).item()),
+    }
+
+
+def _debug_save_latent(
+    x: torch.Tensor,
+    path: Path,
+) -> None:
+    """Save a detached CPU copy without changing the live tensor."""
+    path.parent.mkdir(parents=True, exist_ok=True)
+    torch.save(x.detach().cpu(), path)
+
+
+def _debug_transition(
+    transition_index: int,
+    x_before: torch.Tensor,
+    expanded_raw: torch.Tensor,
+    x_after: torch.Tensor,
+    *,
+    s_i: float,
+    s_next: float,
+    t: float,
+    t_tilde: float,
+    transform: str,
+    seed: int,
+    r: float,
+    kappa_value: float,
+) -> None:
+    """Dump complete state around one SPEED resolution transition."""
+    root = _debug_dir() / f"transition_{transition_index:02d}"
+    root.mkdir(parents=True, exist_ok=True)
+
+    _debug_save_latent(x_before, root / "before.pt")
+    _debug_save_latent(expanded_raw, root / "expanded_raw.pt")
+    _debug_save_latent(x_after, root / "after.pt")
+
+    metadata = {
+        "transition_index": transition_index,
+        "transform": transform,
+        "seed": seed,
+        "source_scale": s_i,
+        "target_scale": s_next,
+        "scale_ratio": r,
+        "t": t,
+        "t_tilde": t_tilde,
+        "kappa": kappa_value,
+        "before": _latent_stats(x_before),
+        "expanded_raw": _latent_stats(expanded_raw),
+        "after": _latent_stats(x_after),
+    }
+
+    with open(root / "metadata.json", "w", encoding="utf-8") as f:
+        json.dump(metadata, f, indent=2)
+
+    print(
+        "\n"
+        "========== SPEED DEBUG TRANSITION ==========\n"
+        f"transition : {transition_index}\n"
+        f"transform  : {transform}\n"
+        f"seed       : {seed}\n"
+        f"scale      : {s_i} -> {s_next}  (r={r:.6f})\n"
+        f"t          : {t:.8f}\n"
+        f"t_tilde    : {t_tilde:.8f}\n"
+        f"kappa      : {kappa_value:.8f}\n"
+        f"before     : {tuple(x_before.shape)}\n"
+        f"expanded   : {tuple(expanded_raw.shape)}\n"
+        f"after      : {tuple(x_after.shape)}\n"
+        f"debug dir  : {root}\n"
+        "\n"
+        f"before stats:\n"
+        f"  min={metadata['before']['min']:.8g} "
+        f"max={metadata['before']['max']:.8g} "
+        f"mean={metadata['before']['mean']:.8g} "
+        f"std={metadata['before']['std']:.8g} "
+        f"rms={metadata['before']['rms']:.8g}\n"
+        "\n"
+        f"raw expansion stats:\n"
+        f"  min={metadata['expanded_raw']['min']:.8g} "
+        f"max={metadata['expanded_raw']['max']:.8g} "
+        f"mean={metadata['expanded_raw']['mean']:.8g} "
+        f"std={metadata['expanded_raw']['std']:.8g} "
+        f"rms={metadata['expanded_raw']['rms']:.8g}\n"
+        "\n"
+        f"after kappa stats:\n"
+        f"  min={metadata['after']['min']:.8g} "
+        f"max={metadata['after']['max']:.8g} "
+        f"mean={metadata['after']['mean']:.8g} "
+        f"std={metadata['after']['std']:.8g} "
+        f"rms={metadata['after']['rms']:.8g}\n"
+        "============================================\n"
+    )
+
+
 # =============================================================================
 # Per-model power-spectrum presets.
 # =============================================================================

@@ -60,6 +123,7 @@ def _expand_and_align_torch(
     H_full: int,
     W_full: int,
 ) -> Tuple[torch.Tensor, float]:
     """Expand a 4D image latent or 5D video latent over its spatial axes."""
+
     if transform not in ("dct", "dwt", "fft"):
         raise ValueError(f"transform must be dct|dwt|fft, got {transform!r}")

     r = s_next / s_i
@@ -77,17 +141,28 @@ def _expand_and_align_torch(
     else:
         raise ValueError(f"expected 4D or 5D latent, got shape {tuple(x.shape)}")

     x_np = x4.detach().cpu().float().numpy()

     if transform == "dwt":
         if abs(r - 2.0) > 1e-6:
             raise ValueError(
                 f"DWT requires r=2 between consecutive scales; got r={r:.4f}."
                 "Use transform=dct or transform=fft for non-dyadic ratios."
             )
         expanded = _dwt_expand_np(x_np, t, seed)
     elif transform == "dct":
         expanded = _dct_expand_np(x_np, (H_tgt, W_tgt), t, seed)
     else:
         expanded = _fft_expand_np(x_np, (H_tgt, W_tgt), t, seed)

+    # Keep a copy before kappa scaling. This is the actual result of the
+    # spectral expansion itself.
+    expanded_raw = torch.from_numpy(expanded)
+
     rescaled = (kappa(t, r) * expanded).astype(np.float32)
     x4_new = torch.from_numpy(rescaled).to(device=x.device, dtype=x.dtype)

     if x.ndim == 5:
         out = x4_new.reshape(B, T_frames, C, H_tgt, W_tgt).permute(0, 2, 1, 3, 4)
     else:
         out = x4_new

+    if _debug_enabled():
+        kappa_value = kappa(t, r)
+        t_tilde = align_timestep(t, r)
+        _debug_transition(
+            transition_index=_debug_transition.counter,
+            x_before=x,
+            expanded_raw=(
+                expanded_raw.reshape(B, T_frames, C, H_tgt, W_tgt)
+                .permute(0, 2, 1, 3, 4)
+                if x.ndim == 5
+                else expanded_raw
+            ).to(dtype=x.dtype),
+            x_after=out,
+            s_i=s_i,
+            s_next=s_next,
+            t=t,
+            t_tilde=t_tilde,
+            transform=transform,
+            seed=seed,
+            r=r,
+            kappa_value=kappa_value,
+        )
+        _debug_transition.counter += 1
+
     return out, align_timestep(t, r)
+
+
+_debug_transition.counter = 0


 # =============================================================================
 # Initial coarse-resolution latent.
 # =============================================================================
```

### One small correction I'd make before applying that literally

Because the function can be called repeatedly during separate ComfyUI executions, I'd reset the counter at the beginning of `sample_speed()` rather than relying on the Python process lifetime.

Add this immediately after entering `sample_speed()`:

```python
if _debug_enabled():
    _debug_transition.counter = 0
```

So each generation gets:

```text
transition_00/
transition_01/
transition_02/
...
```

rather than continuing the numbering from the previous generation.

---

## Running it

Set:

```bash
export SPEED_DEBUG=1
export SPEED_DEBUG_DIR=/media/luna/work/cache/speed-debug
```

then start ComfyUI normally.

For your likely single transition:

```text
1200×800 → 1920×1000
```

you should get something like:

```text
speed-debug/
└── transition_00/
    ├── before.pt
    ├── expanded_raw.pt
    ├── after.pt
    └── metadata.json
```

The important distinction is:

### `before.pt`

This is the **actual Krea2 latent immediately before SPEED expansion**.

### `expanded_raw.pt`

This is:

```text
DCT/DWT/FFT expansion
        ↓
inverse transform
```

**before `kappa()`**.

### `after.pt`

This is the actual tensor that gets handed back into the sampler:

```text
expanded_raw * kappa(t, r)
```

That distinction matters because the current SPEED code does indeed perform the `kappa` rescaling after spectral expansion and then adjusts the flow-matching timestep to `t_tilde`.

---

## One thing I particularly want from your real Krea2 run

The `metadata.json` will let us see something much more useful than just "the latent got bigger."

For example, if you get:

```json
{
  "source_scale": 0.5,
  "target_scale": 1.0,
  "scale_ratio": 2.0,
  "t": 0.73,
  "t_tilde": 0.59,
  "kappa": 1.41,
  "before": {
    "shape": [1, 16, 100, 150],
    "std": 0.91
  },
  "expanded_raw": {
    "shape": [1, 16, 200, 300],
    "std": 0.64
  },
  "after": {
    "shape": [1, 16, 200, 300],
    "std": 0.90
  }
}
```

we can see exactly how SPEED is manipulating the signal statistics.

And **`expanded_raw.pt` is the really juicy one** for your dual-GPU idea.

We can take that tensor and independently inspect:

- spectral energy distribution
- spatial autocorrelation
- how much of the expanded state is inherited vs newly stochastic
- whether the new information is actually spatially independent
- what happens at the eventual GPU partition boundary

That gives us a concrete tensor to experiment with instead of theorizing about what SPEED *probably* does.

Also, one correction to my previous answer: I described the DCT expansion as simply putting the source coefficients in the upper-left and filling everything else with noise. That's indeed what this implementation does. The FFT version instead centers the original spectrum and fills the outer spectrum with complex noise. The DWT version is particularly clean: source becomes LL and fresh noise becomes LH/HL/HH.

So **dumping all three states around the transition is exactly the right instrumentation point**.