# Mage-VL-Cowork speed bench on MLX (mlx-vlm). Usage: mage_bench.py <mlx_model_dir> <label> <outdir>
import sys, os, json, time, random, statistics as st, subprocess, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import filler_text as S
from common import *
import mlx.core as mx, mlx_vlm
MP, LABEL, OUT = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # screens/ paths are relative to this folder
T0 = time.time()
def log(*a): print(f"[{time.strftime('%H:%M:%S')} {time.time()-T0:6.0f}s]", *a, flush=True)
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
env = {"label": LABEL, "model_path": os.path.basename(MP), "chip": sh("sysctl -n machdep.cpu.brand_string"), "ram_gb": int(sh("sysctl -n hw.memsize"))//2**30,
       "macos": sh("sw_vers -productVersion")+" ("+sh("sw_vers -buildVersion")+")", "mlx": mx.__version__, "mlx_vlm": mlx_vlm.__version__,
       "python": platform.python_version(), "power": sh("pmset -g batt | head -1"), "therm_start": sh("pmset -g therm"),
       "size_gb": round(sum(os.path.getsize(os.path.join(MP,f)) for f in os.listdir(MP) if f.endswith(".safetensors"))/1e9, 3),
       "date": time.strftime("%Y-%m-%d %H:%M %Z")}
tl = time.time(); model, processor = load(MP); env["load_s"] = round(time.time()-tl, 1); log(env)
tok = processor.tokenizer
EOS = [151645, 151643]
def no_eos(tokens, logits):
    logits[..., EOS] = -float("inf"); return logits
def timed(prompt, images=None, max_tokens=256, ban_eos=True):
    mx.reset_peak_memory(); mx.clear_cache()
    t0 = time.perf_counter(); ttft = None; text = ""; last = None; marks = {}
    kw = dict(max_tokens=max_tokens, temperature=0.0, seed=0)
    if ban_eos: kw["logits_processors"] = [no_eos]
    for r in stream_generate(model, processor, prompt, image=images, **kw):
        now = time.perf_counter() - t0
        if ttft is None: ttft = now
        text += r.text; last = r
        if "think_end" not in marks and "</think>" in text: marks["think_end"] = now
        if "action_line" not in marks and "Action:" in text: marks["action_line"] = now
        if "tool_call_done" not in marks and "</tool_call>" in text: marks["tool_call_done"] = now
    total = time.perf_counter() - t0
    return dict(epoch_start=time.time()-total, epoch_end=time.time(), ttft_s=ttft, total_s=total, prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens,
                prompt_tps=last.prompt_tps, gen_tps=last.generation_tps, peak_mem_gb=mx.get_peak_memory()/1e9, finish=last.finish_reason, text=text, **{f"t_{k}_s": v for k, v in marks.items()})
def med(runs, keys): return {k: st.median(r[k] for r in runs if r.get(k) is not None) for k in keys if any(r.get(k) is not None for r in runs)}
# ---- text perf (filler text from filler_text.py, no system prompt, thinking default on) ----
random.seed(S.FILLER_SEED)
def sentence(): return " ".join(random.choice(S.FILLER_WORDS) for _ in range(random.randint(8,16))).capitalize()+"."
FILLER = " ".join(sentence() for _ in range(12000)); FIDS = tok.encode(FILLER, add_special_tokens=False)
def chat_text(n): return render(processor, [{"role":"user","content":S.PERF_HEAD + tok.decode(FIDS[:n])}])
def fit(target):
    n = target
    for _ in range(5):
        p = chat_text(n); L = len(tok.encode(p, add_special_tokens=False))
        if abs(L-target) <= 8: break
        n = max(16, n + (target-L))
    return p
KEYS = ["ttft_s","total_s","prompt_tokens","gen_tokens","prompt_tps","gen_tps","peak_mem_gb","t_think_end_s","t_action_line_s","t_tool_call_done_s"]
res = {"env": env, "text_perf": [], "image_turn": []}
def save(): json.dump(res, open(os.path.join(OUT, "bench.json"), "w"), indent=2)
for ctx in S.PERF_CONTEXTS:
    p = fit(ctx)
    try:
        timed(p); runs = [timed(p) for _ in range(3)]
    except Exception as e:
        log("FAIL", ctx, repr(e)); res["text_perf"].append({"ctx": ctx, "error": repr(e)}); save(); break
    row = {"ctx": ctx, **med(runs, KEYS), "spread_gen_tps": [round(r["gen_tps"],2) for r in runs], "spread_prompt_tps": [round(r["prompt_tps"],1) for r in runs], "runs": runs}
    res["text_perf"].append(row); save()
    log(f"TEXT ctx={ctx} n={row['prompt_tokens']} pp={row['prompt_tps']:.1f} tg={row['gen_tps']:.2f} ttft={row['ttft_s']:.3f} peak={row['peak_mem_gb']:.2f} spread_tg={row['spread_gen_tps']}")
# ---- image + text: one full agent turn, official system prompt + first-step user format, natural stop ----
for name, img, instr in [("settings_1280x800", "screens/settings.png", "Turn on Dark Mode."),
                         ("settings_1920x1080", "screens/settings_1080p.png", "Turn on Dark Mode.")]:
    p = render(processor, first_msgs(instr)); j = jpeg95(img)
    timed(p, [j], max_tokens=4096, ban_eos=False); runs = [timed(p, [j], max_tokens=4096, ban_eos=False) for _ in range(3)]
    row = {"name": name, "image": img, "instruction": instr, **med(runs, KEYS), "spread_total_s": [round(r["total_s"],2) for r in runs], "runs": runs}
    res["image_turn"].append(row); save()
    log(f"IMG {name} n={row['prompt_tokens']} ttft={row['ttft_s']:.2f} action={row.get('t_action_line_s')} toolcall={row.get('t_tool_call_done_s')} total={row['total_s']:.2f} gen={row['gen_tokens']} tg={row['gen_tps']:.1f} pp={row['prompt_tps']:.0f} peak={row['peak_mem_gb']:.2f}")
    log("   ->", runs[0]["text"].split("</think>")[-1].strip().replace("\n"," ")[:300])
env["therm_end"] = sh("pmset -g therm"); env["elapsed_s"] = round(time.time()-T0,1); save(); log("DONE")
