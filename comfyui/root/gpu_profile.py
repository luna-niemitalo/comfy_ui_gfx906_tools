# gpu_profile.py
import os
import logging
from contextlib import nullcontext

PROFILE_ENABLED = os.environ.get("COMFY_PROFILE", "0") == "1"

_prof = None

def start_profiler():
    global _prof
    if not PROFILE_ENABLED:
        return nullcontext()

    from torch.profiler import profile, ProfilerActivity

    logging.info("torch.profiler enabled (COMFY_PROFILE=1) — this adds overhead, "
                 "use for diagnosis runs only, not normal usage.")

    _prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,   # lets you jump from op -> source line later
    )
    _prof.__enter__()
    return _prof


def dump_profiler():
    if _prof is None:
        return
    _prof.__exit__(None, None, None)

    # Text table sorted by actual device time — this is your ranked hot-op list
    table = _prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    with open("profile_report.txt", "w") as f:
        f.write(table)
    logging.info("Wrote profile_report.txt")

    # Chrome trace — open in chrome://tracing or https://ui.perfetto.dev
    # for a visual timeline, this is where "browsing" the run visually happens
    _prof.export_chrome_trace("profile_trace.json")
    logging.info("Wrote profile_trace.json (open in chrome://tracing or ui.perfetto.dev)")