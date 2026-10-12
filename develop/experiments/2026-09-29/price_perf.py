"""Price vs Krea2 throughput (2560x1440, 8 steps, sampling only) -> develop/out/price_perf.png.
Measured: Radeon VII rows (this project). Everything else ESTIMATED: step times from spec-sheet
compute x typical efficiency, prices rough 2026 EUR (used where a used market exists; Strix
systems are whole computers). Error bars show the estimate ranges."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

STEPS = 8
# name, price EUR (lo, hi), step seconds (lo, hi), watts, measured
DEVICES = [
    ("2x Radeon VII (yours: 400 + 200 €)", (600, 600), (10.9, 10.9), 600, True),
    ("1x Radeon VII (used today)", (200, 200), (20.0, 20.0), 300, True),
    ("RTX 3090 (used)", (550, 750), (5.0, 8.0), 350, False),
    ("RTX 4090 (used)", (1400, 1900), (2.0, 3.0), 450, False),
    ("RTX 5090 (new)", (2300, 2900), (1.0, 1.5), 575, False),
    ("Strix Halo 128 GB (whole system)", (1900, 2600), (12.0, 25.0), 120, False),
    ("Strix Point laptop (whole system)", (1200, 1800), (40.0, 70.0), 45, False),
]
COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#7f7f7f"]
OUT = "/media/luna/work/tinker/ComfyUI/develop/out/price_perf.png"


def per_hour(step_s):
    return 3600 / (step_s * STEPS)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.5))
for (name, price, step, watts, measured), color in zip(DEVICES, COLORS):
    p_mid = sum(price) / 2
    ih_lo, ih_hi = per_hour(step[1]), per_hour(step[0])
    ih_mid = per_hour(sum(step) / 2)
    ax1.errorbar(p_mid, ih_mid, xerr=[[p_mid - price[0]], [price[1] - p_mid]], yerr=[[ih_mid - ih_lo], [ih_hi - ih_mid]],
                 fmt="o", ms=11, color=color, mfc=color if measured else "white", mew=2, capsize=4, label=name)
    value = [100 * per_hour(s) / p for s, p in ((step[1], price[1]), (step[0], price[0]))]
    ax2.barh(name, sum(value) / 2, xerr=[[sum(value) / 2 - value[0]], [value[1] - sum(value) / 2]], color=color,
             alpha=1.0 if measured else 0.45, capsize=4, edgecolor=color)
    wh = [watts * s * STEPS / 3600 for s in step]
    ax3.barh(name, sum(wh) / 2, xerr=[[sum(wh) / 2 - wh[0]], [wh[1] - sum(wh) / 2]], color=color,
             alpha=1.0 if measured else 0.45, capsize=4, edgecolor=color)

for iso in (2, 5, 10, 20):  # images per hour per 100 EUR
    xs = [150, 3200]
    ax1.plot(xs, [iso * x / 100 for x in xs], ls=":", color="gray", lw=0.8)
    ax1.text(3000, iso * 30, f"{iso}/h per 100 €", fontsize=8, color="gray", ha="right", va="bottom")
ax1.set(xscale="log", yscale="log", xlim=(150, 3300), ylim=(5, 600), xlabel="price (EUR, log)",
        ylabel="2560x1440 images per hour (8 steps, log)", title="Price vs throughput")
ticks = [200, 300, 500, 1000, 2000, 3000]
ax1.set_xticks(ticks, [f"{t} €" for t in ticks])
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=8, loc="upper left")
ax2.set(xlabel="images per hour per 100 € (higher = better value)", title="Value")
ax2.invert_yaxis()
ax2.grid(alpha=0.3, axis="x")
ax3.set(xlabel="Wh per image, GPU/package power (lower = better)", title="Energy per image")
ax3.invert_yaxis()
ax3.grid(alpha=0.3, axis="x")
ax3.tick_params(labelleft=False)
fig.suptitle("Krea2 2560x1440, 8 steps, sampling only. Filled = measured (Radeon VII); hollow/faded = estimate "
             "(spec x typical efficiency, rough 2026 prices)", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110)
print(OUT)
