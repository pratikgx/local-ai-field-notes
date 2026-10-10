"""
Benchmark a few open-weight models on this Mac with MLX and log the results.

Run from the repo root:
    bash scripts/bench.sh

What it does:
  1. Reads your chip, memory and macOS version.
  2. Picks 2 to 4 models that fit your memory (4-bit MLX builds from Hugging Face).
  3. Runs each one with the same prompt and records speed and peak memory.
  4. Saves results/<date>.json, fills the table in models.md, and writes the
     machine section of today's note in notes/.

Pause anything else that uses the GPU (including Darkbloom) while it runs,
or the speeds will be lower than your Mac can really do.
"""
import datetime, json, os, platform, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = "Explain in three short paragraphs why someone might run AI models on their own computer."
MAX_TOKENS = 256

# (min memory in GB, model repo on Hugging Face, plain name, approx download GB)
CANDIDATES = [
    (8,  "mlx-community/gemma-3-4b-it-qat-4bit", "Gemma 3 4B", 3.0),
    (16, "mlx-community/Qwen3-8B-4bit", "Qwen3 8B", 4.6),
    (16, "mlx-community/gemma-3-12b-it-qat-4bit", "Gemma 3 12B", 8.0),
    (24, "lmstudio-community/Qwen3-14B-MLX-4bit", "Qwen3 14B", 8.3),
    (32, "mlx-community/gemma-3-27b-it-qat-4bit", "Gemma 3 27B", 16.9),
    (32, "lmstudio-community/Qwen3-30B-A3B-MLX-4bit", "Qwen3 30B-A3B (MoE)", 17.2),
]
# For machines with 48 GB or more, test a spread: fast, mid, dense large, MoE large.
BIG_SET = ["Qwen3 8B", "Qwen3 14B", "Gemma 3 27B", "Qwen3 30B-A3B (MoE)"]

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""

def machine():
    chip = sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    mem_bytes = sh(["sysctl", "-n", "hw.memsize"])
    mem_gb = round(int(mem_bytes) / 1024**3) if mem_bytes.isdigit() else 0
    macos = sh(["sw_vers", "-productVersion"])
    model_id = sh(["sysctl", "-n", "hw.model"])
    return {"chip": chip, "memory_gb": mem_gb, "macos": macos, "hw_model": model_id}

def run_model(repo):
    cmd = [sys.executable, "-m", "mlx_lm", "generate", "--model", repo,
           "--prompt", PROMPT, "--max-tokens", str(MAX_TOKENS)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"error": "timed out"}
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        # older mlx-lm versions use the mlx_lm.generate module path
        cmd[2:4] = ["mlx_lm.generate"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        out = p.stdout + "\n" + p.stderr
        if p.returncode != 0:
            return {"error": out.strip().splitlines()[-1][:200] if out.strip() else "failed"}
    gen = re.search(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", out)
    pr = re.search(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec", out)
    mem = re.search(r"Peak memory:\s*([\d.]+)\s*GB", out)
    text = out.split("==========")[1].strip() if out.count("==========") >= 2 else ""
    return {
        "gen_tokens": int(gen.group(1)) if gen else None,
        "gen_tps": float(gen.group(2)) if gen else None,
        "prompt_tps": float(pr.group(2)) if pr else None,
        "peak_mem_gb": float(mem.group(1)) if mem else None,
        "sample": text[:400],
    }

def main():
    m = machine()
    today = datetime.date.today().isoformat()
    print(f"Machine: {m['chip']}, {m['memory_gb']} GB, macOS {m['macos']}")
    if not m["memory_gb"]:
        print("Could not read memory size. Is this a Mac?"); sys.exit(1)
    if m["memory_gb"] >= 48:
        picks = [c for c in CANDIDATES if c[2] in BIG_SET]
    else:
        picks = [c for c in CANDIDATES if c[0] <= m["memory_gb"]][-4:]
    total_dl = sum(c[3] for c in picks)
    print(f"Will test {len(picks)} models, about {total_dl:.0f} GB of downloads the first time.\n")
    results = []
    for _, repo, name, _dl in picks:
        print(f"Running {name} ({repo}) ...", flush=True)
        r = run_model(repo)
        r.update({"model": name, "repo": repo})
        results.append(r)
        if r.get("error"):
            print(f"  failed: {r['error']}\n")
        else:
            print(f"  {r['gen_tps']} tokens/sec, peak memory {r['peak_mem_gb']} GB\n")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{today}.json"), "w") as f:
        json.dump({"date": today, "machine": m, "prompt": PROMPT, "max_tokens": MAX_TOKENS, "results": results}, f, indent=2)

    # models.md table
    rows = []
    for r in results:
        if r.get("error"):
            continue
        rows.append(f"| {today} | [{r['model']}](https://huggingface.co/{r['repo']}) | 4-bit MLX | mlx-lm | "
                    f"{r['peak_mem_gb']} GB | {r['gen_tps']:.1f} | [your verdict] |")
    mpath = os.path.join(ROOT, "models.md")
    md = open(mpath).read()
    md = md.replace("| | | | | | | |\n", "")
    marker = "| --- | --- | --- | --- | --- | --- | --- |\n"
    md = md.replace(marker, marker + "\n".join(rows) + ("\n" if rows else ""), 1)
    md = re.sub(r"I measure speed by \[.*?\]\.",
                f"I measure speed with mlx-lm's own numbers, generating {MAX_TOKENS} tokens from the same prompt for every model. "
                "Speed is generation tokens per second. Memory is peak memory reported by mlx-lm. "
                "The script is in scripts/bench.py and raw results are in results/.", md)
    open(mpath, "w").write(md)

    # machine section for today's note
    lines = [f"Chip {m['chip']}, {m['memory_gb']} GB unified memory, macOS {m['macos']} ({m['hw_model']}).", ""]
    ok = [r for r in results if not r.get("error")]
    if ok:
        best = max(ok, key=lambda r: r["gen_tps"] or 0)
        big = max(ok, key=lambda r: r["peak_mem_gb"] or 0)
        lines.append(f"Fastest was {best['model']} at {best['gen_tps']:.0f} tokens per second. "
                     f"The biggest model I ran was {big['model']}, which peaked at {big['peak_mem_gb']} GB of memory "
                     f"and generated {big['gen_tps']:.0f} tokens per second.")
    with open(os.path.join(ROOT, "notes", f"{today}-auto-machine.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Done. Updated models.md, wrote results/ and notes/" + f"{today}-auto-machine.md")
    print("Paste results/" + f"{today}.json back to Claude to write the note and the post.")

if __name__ == "__main__":
    main()
