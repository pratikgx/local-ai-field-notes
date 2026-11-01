import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "2026-10-30-mage-vl-cowork")
IMG = os.path.join(ROOT, "notes", "img")
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = json.load(open(RES + ".json"))["builds"]
b4 = r["mlx-4bit-g64 (LM 4-bit affine g64, vision bf16)"]; b16 = r["mlx-bf16"]
BG, FG, MUTED = "#000000", "#e8e6e3", "#8a8580"; C4, C16 = "#c9a227", "#6f8fb3"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), facecolor=BG, gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
labels = ["512", "4K", "16K", "32K"]; xs = range(4)
def style(ax, title):
    ax.set_facecolor(BG); [s.set_visible(False) for s in ax.spines.values()]
    ax.grid(axis="y", color="#222", lw=0.8); ax.set_axisbelow(True); ax.set_title(title, color=FG, fontsize=13, loc="left", pad=12)
for ax, key, title in [(axes[0], "gen_tps", "Decode speed (tok/s)"), (axes[1], "prompt_tps", "Prefill speed (tok/s)")]:
    style(ax, title)
    for b, c, name in [(b4, C4, "4-bit MLX"), (b16, C16, "BF16 MLX")]:
        ys = [t[key] for t in b["text"]]; ax.plot(xs, ys, color=c, lw=2.6, marker="o", ms=7, label=name)
        for x, y in zip(xs, ys): ax.annotate(f"{y:,.0f}" if key == "prompt_tps" else f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=c, fontsize=10)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels); ax.set_xlabel("context (tokens)"); ax.set_ylim(0, None)
    ax.margins(y=0.18)
axes[0].legend(frameon=False, labelcolor=FG, loc="upper right")
ax = axes[2]; style(ax, "One agent turn from a screenshot (s)")
names = ["1280x800\n4-bit", "1920x1080\n4-bit", "1280x800\nBF16", "1920x1080\nBF16"]
turns = b4["image_turn"] + b16["image_turn"]; cols = [C4, C4, C16, C16]
ttft = [t["ttft_s"] for t in turns]; done = [t["t_tool_call_done_s"] for t in turns]
ax.bar(range(4), done, color=cols, alpha=0.35, width=0.6, label="tool call ready")
ax.bar(range(4), ttft, color=cols, width=0.6, label="first token")
for i, (a, d) in enumerate(zip(ttft, done)):
    ax.text(i, d + 0.12, f"{d:.1f}s", ha="center", color=FG, fontsize=10); ax.text(i, a / 2, f"{a:.1f}", ha="center", va="center", color=BG, fontsize=9, weight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9); ax.set_ylim(0, max(done) * 1.25)
ax.legend(frameon=False, labelcolor=FG, loc="upper left", fontsize=9)
fig.suptitle("Microsoft Mage-VL-Cowork 4B  ·  M5 Pro Mac mini 64GB  ·  mlx-vlm", color=FG, fontsize=16, x=0.06, ha="left", y=0.99)
fig.text(0.06, 0.015, "Median of 3 runs after 1 warmup, greedy, 256 generated tokens for text. Peak MLX memory: 4-bit 3.9-8.8 GB, BF16 10.0-15.2 GB. Background inference app running during tests.", color=MUTED, fontsize=9, ha="left")
plt.tight_layout(rect=(0.02, 0.04, 1, 0.94)); plt.savefig(os.path.join(IMG, "2026-10-30-mage-speed.png"), dpi=200, facecolor=BG)
