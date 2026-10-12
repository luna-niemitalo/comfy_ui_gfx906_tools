# rvii-comfy

Luna's ComfyUI work for 2x Radeon VII (gfx906): int4 Krea2, fused attention, the two-GPU split,
Resolution Growth, and the experiments and tools around them, all in one repo.
`deploy.nu` installs it into a ComfyUI checkout.

## Layout

| path | what | history |
|---|---|---|
| `nodes/ComfyUI-Gfx906Int4` | int4 GEMM, fused attention, row/batch split, int8 VAE, training kernels | merged with full history |
| `nodes/ComfyUI-ResolutionGrowth` | Resolution Growth nodes (fixed + progressive) | merged with full history |
| `nodes/ComfyUI_Luna_QOL_Nodes` | small QOL nodes (material saver, resolution selector, dual-GPU tile sampler) | starts here |
| `develop/` | harness, recipes, sweeps, dated experiments (`experiments/<date>/`) | merged with full history |
| `oc_tools/` | OC encyclopedia tools (render, batch, sheet, catalog, book) | starts here |
| `comfyui/BASE` | the upstream ComfyUI commit our patches apply to | |
| `comfyui/patches/` | our edits to upstream ComfyUI files, as a `git am` series | from branch `rvii-perf` |
| `comfyui/root/` | files we add to the ComfyUI root: `run.nu`, `nix/`, `.envrc`, `gpu_profile.py`, ... | replayed with history |
| `third_party/nodes.toml` | third-party nodes we use, pinned by commit (not vendored) | |
| `third_party/patches/` | our local fixes to three of them | |
| `CLAUDE.md` | status and resume notes (the old ComfyUI `CLAUDE.md`, with history) | |
| `DESIGN.md` | how the repo and the deploy work, invariants | |

The OC encyclopedia's content (canon, prompt blocks, catalog, finals, references) is its own repo,
`../succubus-oc`. `deploy.nu --oc` links it in.

## Deploy

    nu deploy.nu <comfyui checkout> --dry-run                     # show the plan, change nothing
    nu deploy.nu <comfyui checkout>                               # patches, third-party nodes, missing links
    nu deploy.nu <comfyui checkout> --replace --oc ../succubus-oc # also turn existing real files into links

The checkout must contain `comfyui/BASE`. A fresh one:

    git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI
    git -C ComfyUI checkout -b rvii <contents of comfyui/BASE>
    nu deploy.nu ComfyUI

## Updating

- **Nodes, develop, oc_tools, root files**: edit them through the ComfyUI links or here; it is the
  same file. Commit here.
- **Upstream ComfyUI files**: commit in the ComfyUI checkout, then regenerate the series:
  `git -C <comfy> format-patch -o comfyui/patches <BASE>..HEAD -- comfy main.py requirements.txt .gitignore extra_model_paths.yaml.example`
- **Rebasing onto newer upstream**: rebase the checkout's branch, write the new commit to
  `comfyui/BASE`, regenerate the series.
- **Third-party nodes**: change the pinned commit in `nodes.toml`; refresh a patch with
  `git -C <node> diff > third_party/patches/<node>.patch`.

## Not here

- No remote. This repo exists only on `unreliable_pool/work`; a second copy is Luna's step.
- `.claude/settings.local.json` stays in the ComfyUI checkout (globally gitignored).
- Experiment outputs (`develop/out/`) are gitignored.
