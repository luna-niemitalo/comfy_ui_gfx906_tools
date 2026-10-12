// "Stop (keep result)" on Resolution Growth (Progressive): asks the running node to finish its
// current step and return what it has (ComfyUI's Cancel drops the node's outputs). Server side:
// stop.py, POST /resolution_growth/stop.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "ResolutionGrowth.StopButton",
  nodeCreated(node) {
    if (node.comfyClass !== "ResolutionGrowthProgressive") return;
    node.addWidget("button", "Stop (keep result)", null, async () => {
      const response = await api.fetchApi("/resolution_growth/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(node.id) }),
      });
      if (!response.ok) console.error("Resolution Growth stop failed:", response.status, await response.text());
    }, { serialize: false });
  },
});
