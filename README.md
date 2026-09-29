# ComfyUI-ResolutionGrowth

Render far above a model's training resolution without its high-resolution artifacts
(mirroring, splotches, repeated subjects): compose at the native size, then grow the image to
the target in many small steps. Each step upscales the last clean image a little in pixel
space, re-noises it and runs one denoise in tiles at the model's native size, so attention
never sees more positions than it was trained on. The composition's low frequencies are held
(anchored) while the steps add detail.

Tested on Krea2 (int4, 2x Radeon VII): clean 3840x2160 in ~2 min, 7680x4320 in ~9 min, with
the layout drift from the composition flat from 1.3 to 33 MP.

## Nodes

**Resolution Growth**: fixed target.
- `growth_mode`: `geometric` (`growth_steps` equal-ratio steps, faster) or `linear`
  (+`step_px` of width per step, about 2x slower, cleaner).
- `oversample`: grow to the target x this, then Lanczos down. This gives supersampling,
  with more detail per output pixel.
- Outputs the latent plus a plot of the size, sigma and anchor curves.

**Resolution Growth (Progressive)**: fixed output size, Cycles-style.
- The internal image grows by `step_px` per step, up to `max_oversample` x the output.
- Each step is shown at output size and compared with the previous one.
- Stops when a step changes the output by less than `converge_db`, at the size cap, or on
  **Stop (keep result)**. ComfyUI's Cancel discards the result; this button keeps it.
- Outputs the image, latent, schedule plot and a per-step report.

## Wiring

    loader -> (optional speed-ups / 2-GPU split) -> KSampler at native size (~1 MP)
           -> Resolution Growth -> VAE Decode

## Curves

- sigma and anchor follow the size. `f` is how far a step starts along the growth, measured
  in log-area from 0 to 1.
  - `sigma` runs along the model's own schedule, from `sigma_start` down to `sigma_end`.
  - `anchor` is `anchor * (1 - f) ** anchor_shape`.
- `size_shape` bends geometric growth: above 1 the growth comes late, below 1 it comes early.
- `sigma_end` > 0 keeps adding a little noise in every step. It's untested, meant for
  targets far above native, where pure refinement starts to look like AI upscaling.

## Tips

- `detail_positive`: a prompt for the tiles that describes materials, texture and style. Leave
  out the subject and where it goes, or every tile draws one.
- `tile_size`: keep it at the model's training size.
- `tile_batch`: use an even number with a batch-splitting multi-GPU node.
- ROCm: the growth steps' VAE calls run without MIOpen, because MIOpen rebuilds kernels for
  every new size. The tile loop also syncs after every batch, which works around a ROCm 7.2
  runtime crash.
