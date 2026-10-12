# rvii-comfy design

## Purpose

One repo, owned by Luna, holding every line of our ComfyUI work, so that the ComfyUI checkout is
replaceable: re-clone, upgrade or rebase it, then deploy again.

## Invariants

1. **Upstream stays upstream.** We never vendor ComfyUI or third-party nodes. Our changes to
   upstream files live in `comfyui/patches/`, a `git am` series against `comfyui/BASE`. Third-party
   nodes are pinned in `third_party/nodes.toml`, and our fixes to them live in
   `third_party/patches/`.
2. **One copy of our code.** Our nodes, `develop/`, `shot_tools/` and the root tooling are symlinked
   into the checkout, not copied. Editing through ComfyUI edits the repo.
3. **Deploy never deletes.** A real file where a link belongs is skipped, or with `--replace` moved
   to `<comfy>/.rvii-backup/<timestamp>/`. Runtime data inside a replaced dir (`develop/out`) is
   moved into the repo's gitignored tree first, so code finds it at the same path.
4. **Deploy is idempotent.** A step that is already done reports `ok`. Patches are checked with
   `git apply --reverse --check` before `git am`.
5. **Validate at the boundary.** The target must be a git checkout of a ComfyUI tree that contains
   `comfyui/BASE`. Failures print expected and actual.
6. **Project content is not toolchain.** A project's content, its project-specific experiments
   and tools live in the project's own repo (the OC: canon, prompts, catalog, part-growth and
   character-sheet scripts, encyclopedia tools). This repo holds only what is generic. The shot
   tools are linked one by one into the project's `tools/`, and they resolve paths from where they
   are invoked (`os.path.abspath`, not `realpath`), so `ComfyUI/develop/<project>/tools/render.py`
   still sees `ComfyUI/` as the repo root and `develop/<project>` as the project.
   (2026-10-12: the OC material that first landed here was rewritten out of this repo's history.)

## Rejected alternatives

- **A ComfyUI fork as the repo**: our 35 commits sit on 10k+ upstream ones, and most of our work
  is in nodes and develop/. A fork mixes ours with theirs and drags upstream history along.
- **Git submodules for the nodes**: three histories, three remotes to keep in step. The
  subtree-merged histories keep authorship and dates with one repo.
- **Copy-deploy**: two copies drift. A fix made inside ComfyUI is lost on the next deploy.
- **Vendoring third-party nodes**: their code in our history, and their updates become merges.

## Hazards

- The root tooling (`run.nu`, `nix/`, ...) is also tracked on the old ComfyUI branch `rvii-perf`.
  A checkout on that branch shows those files as deleted after `--replace`. Use a branch made from
  `comfyui/BASE` plus `comfyui/patches` (deploy does the `git am`).
- The patch series covers only `comfy/`, `main.py`, `requirements.txt`, `.gitignore` and
  `extra_model_paths.yaml.example`. An upstream file edited anywhere else will not be picked
  up when the series is regenerated. Extend the path list in README "Updating".
