#!/usr/bin/env python3
"""Draft every shot of one or more shot files through a running render_server.py, one at a time.
Resumable: a shot with a non-rejected draft already in the manifest is skipped. A failed shot is
logged and the batch goes on; a dead server stops the batch (restart it, rerun the same command).

--watch: instead of named files, draft every shots/*.yaml, re-scanning after each pass so newly
written shot files are picked up; idle-waits when nothing is pending and exits after --idle-exit
seconds idle. A failed shot is not retried within the same process.

Usage (from the ComfyUI root, server running; <project> as in render.py):
  direnv exec . python develop/<project>/tools/batch.py develop/<project>/shots/<file>.yaml [...]
  direnv exec . python develop/<project>/tools/batch.py --watch [--idle-exit 3600]
"""
import os
import sqlite3
import sys
import time

import yaml

import render
import render_server


def drafted(image_id):
    with sqlite3.connect(render.MANIFEST) as db:
        db.execute(render.SCHEMA)
        return db.execute("SELECT 1 FROM images WHERE image_id = ? AND status = 'draft'", (image_id,)).fetchone()


def run_pass(shot_files, skip):
    """Draft every pending shot once; returns (drafted, stopped). skip: ids failed earlier."""
    todo = []
    for path in shot_files:
        with open(path) as f:
            todo += [(path, shot["image_id"]) for shot in yaml.safe_load(f)["shots"]]
    pending = [(path, image_id) for path, image_id in todo if image_id not in skip and not drafted(image_id)]
    print(f"[batch] {len(todo)} shots, {len(todo) - len(pending)} drafted or skipped, {len(pending)} to go", flush=True)
    done = 0
    for n, (path, image_id) in enumerate(pending, 1):
        if drafted(image_id):  # another client got there first
            continue
        try:
            reply = render_server.ask({"shots": path, "id": image_id, "seed": None, "final": False})
        except ConnectionError as e:
            print(f"[batch] STOP at {image_id}: {e}", flush=True)
            return done, True
        if "error" in reply:
            skip.add(image_id)
            print(f"[batch] {n}/{len(pending)} {image_id} FAILED: {reply['error'].strip().splitlines()[-1]}", flush=True)
            continue
        done += 1
        print(f"[batch] {n}/{len(pending)} {image_id} run {reply['ok']['run_id']} "
              f"{reply['ok']['timings']['total_s']} s", flush=True)
    return done, False


def main():
    args = sys.argv[1:]
    watch = "--watch" in args
    idle_exit = 3600
    if "--idle-exit" in args:
        idle_exit = int(args[args.index("--idle-exit") + 1])
    if not watch:
        shot_files = [os.path.abspath(p) for p in args]
        if not shot_files:
            sys.exit(__doc__)
        skip = set()
        t_start = time.perf_counter()
        run_pass(shot_files, skip)
        print(f"[batch] done in {time.perf_counter() - t_start:.0f} s, failed: {sorted(skip)}", flush=True)
        return
    shots_dir = os.path.join(render.PROJECT_DIR, "shots")
    skip, idle_since = set(), time.time()
    while True:
        shot_files = sorted(os.path.join(shots_dir, n) for n in os.listdir(shots_dir) if n.endswith(".yaml"))
        done, stopped = run_pass(shot_files, skip)
        if stopped:
            return
        if done:
            idle_since = time.time()
        elif time.time() - idle_since > idle_exit:
            print(f"[batch] idle {idle_exit} s, exiting; failed: {sorted(skip)}", flush=True)
            return
        else:
            time.sleep(60)

if __name__ == "__main__":
    main()
