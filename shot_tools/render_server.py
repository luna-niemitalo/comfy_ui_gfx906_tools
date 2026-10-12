#!/usr/bin/env python3
"""render.py with the pipeline kept loaded (saves ~20 s of model load per image).

  serve: load oc_krea2's loaders once, then render requests one at a time from a Unix socket
         (<project>/out/render.sock; machine-local, no network). Shot files and prompt blocks
         are re-read per request; a change to the recipe's loaders needs a server restart.
  send : same arguments as render.py; the server renders and replies with render.py's result line.

Usage (from the ComfyUI root; serve in the background, log next to the drafts; <project> as in render.py):
  direnv exec . python develop/<project>/tools/render_server.py serve
  direnv exec . python develop/<project>/tools/render_server.py send \\
      --shots develop/<project>/shots/<file>.yaml --id <image_id> [--seed N] [--final]
"""
import argparse
import json
import os
import socket
import sys
import time
import traceback

import render

SOCKET_PATH = os.path.join(render.DRAFT_DIR, "render.sock")


def handle(request, loaded):
    # render.py reports bad shots / blocks with sys.exit(message); here that must not end the server
    try:
        shot, parts, seed = render.prepare(request["shots"], request["id"], request["seed"], request.get("from_run"))
        return {"ok": render.run_shot(shot, parts, seed, request["final"], loaded)}
    except SystemExit as e:
        return {"error": str(e)}
    except Exception:
        return {"error": traceback.format_exc()}


def serve():
    render.bootstrap_comfy()  # puts develop/harness (pipeline) on sys.path
    import torch
    import pipeline
    t0 = time.perf_counter()
    with torch.inference_mode():
        loaded = pipeline.build_loaders(pipeline.resolve({"kind": "oc_krea2"}))
    os.makedirs(render.DRAFT_DIR, exist_ok=True)
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)  # left by a crashed server
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(1)
    print(f"[render server] loaded in {time.perf_counter() - t0:.1f} s, listening on {SOCKET_PATH}", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            with conn, conn.makefile("rw") as stream:
                request = json.loads(stream.readline())
                print(f"[render server] request {request}", flush=True)
                reply = handle(request, loaded)
                print(f"[render server] reply {reply}", flush=True)
                stream.write(json.dumps(reply) + "\n")
                stream.flush()
    finally:
        server.close()
        os.unlink(SOCKET_PATH)


def ask(request):
    """Reply dict {"ok": ...} | {"error": ...}; ConnectionError when the server is gone."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(SOCKET_PATH)
    except (FileNotFoundError, ConnectionRefusedError) as e:
        raise ConnectionError(f"no render server listening on {SOCKET_PATH} (start: render_server.py serve)") from e
    with client, client.makefile("rw") as stream:
        stream.write(json.dumps(request) + "\n")
        stream.flush()
        line = stream.readline()
    if not line:
        raise ConnectionError("render server closed the connection without a reply (crashed? see its log)")
    return json.loads(line)


def send(request):
    try:
        reply = ask(request)
    except ConnectionError as e:
        sys.exit(str(e))
    if "error" in reply:
        sys.exit(reply["error"])
    print(json.dumps(reply["ok"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    send_parser = sub.add_parser("send")
    send_parser.add_argument("--shots", required=True)
    send_parser.add_argument("--id", required=True, help="image_id of the shot")
    send_parser.add_argument("--seed", type=int, help="override the shot's seed")
    send_parser.add_argument("--final", action="store_true")
    send_parser.add_argument("--from-run", type=int, help="reuse that manifest run's prompt and seed")
    cli = parser.parse_args()

    if cli.command == "serve":
        serve()
        return
    send({"shots": os.path.abspath(cli.shots), "id": cli.id, "seed": cli.seed, "final": cli.final,
          "from_run": cli.from_run})


if __name__ == "__main__":
    main()
