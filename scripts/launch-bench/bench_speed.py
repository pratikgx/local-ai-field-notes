# Usage: bench_speed.py <model_dir> <tag>
import sys, os, time, statistics as st
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from common import S, ROOT, log, save, done
model_dir, tag = sys.argv[1], sys.argv[2]
out = f"speed_{tag}.json"
if done(out): log("skip", out); sys.exit(0)
cfg = S["speed"]
model, tok = load(model_dir)
tok.eos_token_ids = set()  # ignore EOS so every run generates exactly gen_tokens
maxctx = 16384
text = open(os.path.join(ROOT, "data", "pride_and_prejudice.txt"), encoding="utf-8").read()
ids = tok.encode(text, add_special_tokens=False)
bos = tok.bos_token_id
sampler = make_sampler(0.0)
def one(n):
    prompt = [bos] + ids[: n - 1]
    mx.clear_cache(); mx.reset_peak_memory()
    t0 = time.perf_counter(); ttft = None; last = None
    for r in stream_generate(model, tok, prompt, max_tokens=cfg["gen_tokens"], sampler=sampler,
                             prefill_step_size=cfg["prefill_step_size"]):
        if ttft is None: ttft = time.perf_counter() - t0
        last = r
    wall = time.perf_counter() - t0
    return dict(prompt_tokens=last.prompt_tokens, gen_tokens=last.generation_tokens,
                prefill_tps=last.prompt_tps, decode_tps=last.generation_tps,
                ttft_s=ttft, wall_s=wall, peak_mem_gb=last.peak_memory)
res = {"model_dir": model_dir, "tag": tag, "rows": []}
for n in cfg["prompt_lengths"]:
    n = min(n, maxctx - cfg["gen_tokens"])
    for _ in range(cfg["warmup"]): one(n)
    runs = [one(n) for _ in range(cfg["runs"])]
    agg = {"prompt_len": n, "n_runs": len(runs), "runs": runs}
    for k in ["prefill_tps", "decode_tps", "ttft_s", "peak_mem_gb"]:
        v = [r[k] for r in runs]
        agg[k + "_median"] = st.median(v); agg[k + "_min"] = min(v); agg[k + "_max"] = max(v)
    res["rows"].append(agg)
    log(tag, n, "prefill %.0f tok/s decode %.1f tok/s ttft %.3fs peak %.3f GB" % (
        agg["prefill_tps_median"], agg["decode_tps_median"], agg["ttft_s_median"], agg["peak_mem_gb_median"]))
    save(out + ".partial", res)
save(out, res)
