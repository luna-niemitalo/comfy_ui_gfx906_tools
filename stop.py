"""Stop button for Resolution Growth (Progressive). ComfyUI's Cancel raises inside the node
and drops its outputs; this route instead asks a running progressive node to finish its
current step and return what it has. The button lives in web/resolution_growth.js."""
from aiohttp import web

_requests = set()


def register_routes():
    """POST /resolution_growth/stop {"node_id": "<id>"}. No-op without a running server
    (the develop/ harness imports nodes headless)."""
    import server
    instance = getattr(server.PromptServer, "instance", None)
    if instance is None:
        return

    @instance.routes.post("/resolution_growth/stop")
    async def stop(request):
        # IN, non-critical: the frontend button's JSON body
        try:
            data = await request.json()
        except ValueError:
            return web.json_response({"error": "expected a JSON body {\"node_id\": \"<id>\"}"}, status=400)
        node_id = data.get("node_id") if isinstance(data, dict) else None
        if not isinstance(node_id, (str, int)) or str(node_id) == "":
            return web.json_response({"error": f"expected {{\"node_id\": \"<id>\"}}, got {data!r}"}, status=400)
        _requests.add(str(node_id))
        return web.json_response({"ok": True, "node_id": str(node_id)})


def _matches(unique_id, node_id):
    # a node inside a subgraph runs as "<outer>:<inner>"; the button knows only its own id
    return unique_id == node_id or unique_id.endswith(":" + node_id)


def requested(unique_id):
    return any(_matches(str(unique_id), r) for r in _requests)


def clear(unique_id):
    _requests.difference_update({r for r in _requests if _matches(str(unique_id), r)})
