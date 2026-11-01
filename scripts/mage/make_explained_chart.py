import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "2026-10-30-mage-vl-cowork")
IMG = os.path.join(ROOT, "notes", "img")
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
R = json.load(open(RES + ".json"))["builds"]
b4 = R["mlx-4bit-g64 (LM 4-bit affine g64, vision bf16)"]; b16 = R["mlx-bf16"]
D = {k: json.load(open(os.path.join(RES, f"{k}-demo.json"))) for k in ("4bit", "bf16")}
BG, PANEL, FG, MUTED, GRID = "#050505", "#0b0b0d", "#ece8e1", "#7d786f", "#1c1c20"
C4, C16 = "#d4a72c", "#6f93bd"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11})
fig = plt.figure(figsize=(16, 10.5), facecolor=BG)
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.18, left=0.105, right=0.975, top=0.84, bottom=0.1)
def style(ax, title, sub):
    ax.set_facecolor(PANEL); [s.set_visible(False) for s in ax.spines.values()]
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, color=FG, fontsize=14, loc="left", pad=26, weight="semibold")
    ax.text(0, 1.03, sub, transform=ax.transAxes, color=MUTED, fontsize=10)

# A: where a move's time goes
ax = fig.add_subplot(gs[0, 0]); style(ax, "Where one agent move's time goes", "Dark Mode screen, seconds. Median of 3 runs after 1 warmup.")
turns = [("4-bit  1280x800", b4["image_turn"][0], C4), ("4-bit  1920x1080", b4["image_turn"][1], C4),
         ("Full  1280x800", b16["image_turn"][0], C16), ("Full  1920x1080", b16["image_turn"][1], C16)]
for i, (name, t, c) in enumerate(turns):
    y = 3 - i; read = t["ttft_s"]; think = t["t_think_end_s"] - read; write = t["total_s"] - t["t_think_end_s"]
    ax.barh(y, read, color="#e8e2d6", height=0.58)
    ax.barh(y, think, left=read, color=c, height=0.58)
    ax.barh(y, write, left=read + think, color=c, alpha=0.45, height=0.58)
    ax.text(read / 2, y, f"{read:.1f}", ha="center", va="center", color=BG, fontsize=10, weight="bold")
    if think > 0.6: ax.text(read + think / 2, y, f"{think:.1f}", ha="center", va="center", color=BG, fontsize=10, weight="bold")
    ax.text(t["total_s"] + 0.1, y, f"{t['total_s']:.1f}s", va="center", color=FG, fontsize=11, weight="semibold")
ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels([n for n, _, _ in turns]); ax.tick_params(axis="y", colors=FG, length=0)
ax.set_xlim(0, 8.2); ax.set_xlabel("seconds"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(color="#e8e2d6", label="reading the screenshot"), Patch(color="#9a9a9a", label="thinking"), Patch(color="#9a9a9a", alpha=0.45, label="writing the action")],
          frameon=False, labelcolor=FG, loc="upper right", fontsize=9.5)

# B: peak memory vs context
ax = fig.add_subplot(gs[0, 1]); style(ax, "Memory grows with context, not precision", "Peak MLX memory (GB) by context length, text prompts.")
xs = [t["ctx"] for t in b4["text"]]; lab = ["512", "4K", "16K", "32K"]
for b, c, name, size in [(b4, C4, "4-bit", b4["env"]["size_gb"]), (b16, C16, "Full precision", b16["env"]["size_gb"])]:
    ys = [t["peak_mem_gb"] for t in b["text"]]
    ax.plot(range(4), ys, color=c, lw=2.8, marker="o", ms=7, label=f"{name} ({size:.1f}GB on disk)")
    for i, y in enumerate(ys): ax.annotate(f"{y:.1f}", (i, y), textcoords="offset points", xytext=(0, 10), ha="center", color=c, fontsize=10)
    ax.plot([3, 3], [ys[0], ys[3]], color=c, lw=0)
g4 = b4["text"][3]["peak_mem_gb"] - b4["text"][0]["peak_mem_gb"]; g16 = b16["text"][3]["peak_mem_gb"] - b16["text"][0]["peak_mem_gb"]
ax.text(1.15, 0.9, f"512 to 32K adds +{g4:.1f}GB (4-bit) and +{g16:.1f}GB (full).\nAt 32K the context outweighs the 3.1GB 4-bit model.", color=MUTED, fontsize=9.5)
ax.set_xticks(range(4)); ax.set_xticklabels(lab); ax.set_xlabel("context (tokens)"); ax.set_ylim(0, 18)
ax.legend(frameon=False, labelcolor=FG, loc="upper left", fontsize=9.5)

# C: pixels cost tokens
ax = fig.add_subplot(gs[1, 0]); style(ax, "Bigger screenshot, longer read", "Time to first token (s). Prompt is 2,140 tokens at 1280x800, 3,180 at 1080p.")
import numpy as np
x = np.arange(2); w = 0.34
r4 = [b4["image_turn"][0]["ttft_s"], b4["image_turn"][1]["ttft_s"]]; r16 = [b16["image_turn"][0]["ttft_s"], b16["image_turn"][1]["ttft_s"]]
for off, vals, c, name in [(-w/2, r4, C4, "4-bit"), (w/2, r16, C16, "Full precision")]:
    ax.bar(x + off, vals, width=w, color=c, label=name)
    for xi, v in zip(x + off, vals): ax.text(xi, v + 0.04, f"{v:.2f}s", ha="center", color=FG, fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"1280x800\n{b4['image_turn'][0]['prompt_tokens']:,} tokens", f"1920x1080\n{b4['image_turn'][1]['prompt_tokens']:,} tokens"], color=FG)
ax.set_ylim(0, 2.3); ax.set_ylabel("seconds"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, labelcolor=FG, loc="upper left", fontsize=9.5)

# D: agent runs, step by step
ax = fig.add_subplot(gs[1, 1]); style(ax, "Three tasks, move by move", "Seconds per move, one greedy run each, 1080p. Shell runs in a sandbox, clicks are scored.")
KC = {"screenshot": "#4a4a52", "click": "#e07a5f", "bash": "#81b29a", "terminate": "#e8e2d6", "none": "#2a2a2e"}
rows = []
for task, tname in [("dark_mode", "Dark Mode"), ("csv_total", "CSV total"), ("rename_file", "Rename file")]:
    for k, bname in [("4bit", "4-bit"), ("bf16", "Full")]:
        r = next(r for r in D[k]["results"] if r["task"]["id"] == task); rows.append((f"{tname}  {bname}", r))
for i, (name, r) in enumerate(rows):
    y = len(rows) - 1 - i - (i // 2) * 0.5; left = 0
    for s in r["steps"]:
        calls = s["tool_calls"]
        if not calls: kind = "none"
        elif calls[0].get("name") == "bash": kind = "bash"
        else: kind = calls[0]["arguments"].get("action", "none")
        kind = kind if kind in KC else "click"
        ax.barh(y, s["secs"], left=left, color=KC[kind], height=0.62, edgecolor=PANEL, lw=1.5); left += s["secs"]
    ok = r["verdict"]["correct"]
    note = ("correct" if ok else "missed") if r["verdict"].get("total_reported_to_user", True) else "right sum, never said it"
    if r["verdict"].get("channel") == "gui" and r["task"]["id"] == "rename_file": note = "clicked (first click on target)"
    ax.text(left + 0.8, y, f"{r['task_secs']:.0f}s  {note}", va="center", color=FG if ok else "#e07a5f", fontsize=9.5)
    ax.text(-1, y, name, va="center", ha="right", color=FG, fontsize=10)
ax.set_yticks([]); ax.set_xlim(0, 85); ax.set_xlabel("seconds"); ax.grid(axis="y", visible=False)
ax.legend(handles=[Patch(color=KC[k], label=l) for k, l in [("screenshot", "screenshot"), ("click", "click"), ("bash", "shell command"), ("terminate", "done"), ("none", "no action")]],
          frameon=False, labelcolor=FG, loc="lower right", fontsize=9, ncol=1)

fig.text(0.06, 0.95, "Microsoft Mage-VL-Cowork 4B, explained", color=FG, fontsize=22, weight="bold")
fig.text(0.06, 0.912, "64GB M5 Pro Mac mini  ·  MLX 0.32.4 + mlx-vlm 0.7.11  ·  4-bit language model with full-precision vision vs full precision", color=MUTED, fontsize=12)
fig.text(0.06, 0.018, "Greedy decoding. Text tests: 256 generated tokens. Agent tests use Microsoft's system prompt and HTML mock screens. Background inference app running during tests.", color=MUTED, fontsize=9)
plt.savefig(os.path.join(IMG, "2026-10-30-mage-explained.png"), dpi=180, facecolor=BG)
