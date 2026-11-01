# Builds results/2026-10-30-mage-vl-cowork.json and .csv from the raw files in results/2026-10-30-mage-vl-cowork/.
# Usage, from the repo root: python3 scripts/mage/summarize.py
import json, csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "results", "2026-10-30-mage-vl-cowork")
OUT = os.path.join(ROOT, "results", "2026-10-30-mage-vl-cowork")
W = json.load(open(os.path.join(RAW, "weight_bytes.json")))
HW = {"chip": "Apple M5 Pro", "machine": "Mac mini", "ram_gb": 64, "peak_bandwidth_gbs": 307,
      "source": "https://www.apple.com/mac-mini/specs/", "source_alt": "https://support.apple.com/en-us/128108", "recorded": "2026-10-30"}
KV_PER_TOK = 36 * 8 * 128 * 2 * 2  # layers x kv_heads x head_dim x (k,v) x bf16 bytes; all 36 layers full attention
rows = []; out = {"hardware": HW, "kv_bytes_per_token": KV_PER_TOK, "builds": {}}
for build, d in [("mlx-4bit-g64 (LM 4-bit affine g64, vision bf16)", "4bit"), ("mlx-bf16", "bf16")]:
    b = json.load(open(os.path.join(RAW, f"{d}-bench.json"))); mdir = "mlx_4bit" if d == "4bit" else "mlx_bf16"
    wb = W[mdir]["per_token_decode_bytes"]; ob = {"env": b["env"], "text": [], "image_turn": []}
    for r in b["text_perf"]:
        if "error" in r: ob["text"].append(r); continue
        ctx_avg = r["prompt_tokens"] + r["gen_tokens"] / 2
        bpt = wb + KV_PER_TOK * ctx_avg; gbs = bpt * r["gen_tps"] / 1e9
        row = {"ctx": r["ctx"], "prompt_tokens": r["prompt_tokens"], "gen_tokens": r["gen_tokens"], "prompt_tps": round(r["prompt_tps"],1), "gen_tps": round(r["gen_tps"],2),
               "ttft_s": round(r["ttft_s"],3), "peak_mem_gb": round(r["peak_mem_gb"],2), "spread_gen_tps": r["spread_gen_tps"], "spread_prompt_tps": r["spread_prompt_tps"],
               "bytes_per_token_est": int(bpt), "eff_bw_gbs_est": round(gbs,1), "bw_util_pct_est": round(100*gbs/HW["peak_bandwidth_gbs"],1)}
        ob["text"].append(row)
    for r in b["image_turn"]:
        ob["image_turn"].append({k: (round(v,3) if isinstance(v,float) else v) for k, v in r.items() if k != "runs"} | {
            "spread_ttft_s": [round(x["ttft_s"],2) for x in r["runs"]], "spread_tool_call_done_s": [round(x.get("t_tool_call_done_s") or 0,2) for x in r["runs"]], "reply_text": r["runs"][0]["text"]})
    dm = json.load(open(os.path.join(RAW, f"{d}-demo.json")))
    ob["demo"] = [{"task": x["task"]["id"], "instruction": x["task"]["instruction"], "image": x["task"]["image"], "steps": len(x["steps"]), "task_secs": x["task_secs"],
                   "per_step": [{"step": s["step"], "action_line": s["action_line"], "tool": [c.get("name") for c in s["tool_calls"]],
                                 "first_args": (s["tool_calls"][0].get("arguments") if s["tool_calls"] else None), "secs": s["secs"], "gen_tokens": s["gen_tokens"],
                                 "prompt_tokens": s["prompt_tokens"], "bash_output": s.get("bash_output")} for s in x["steps"]],
                   "verdict": x["verdict"]} for x in dm["results"]]
    out["builds"][build] = ob
json.dump(out, open(OUT + ".json", "w"), indent=1)
with open(OUT + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["build","test","context_tokens","prompt_tps","gen_tps","ttft_s","t_action_line_s","t_tool_call_done_s","total_turn_s","gen_tokens","peak_mlx_mem_gb","eff_bw_gbs_est","bw_util_pct_est"])
    for build, ob in out["builds"].items():
        for r in ob["text"]:
            if "error" in r: continue
            w.writerow([build, "text", r["prompt_tokens"], r["prompt_tps"], r["gen_tps"], r["ttft_s"], "", "", "", r["gen_tokens"], r["peak_mem_gb"], r["eff_bw_gbs_est"], r["bw_util_pct_est"]])
        for r in ob["image_turn"]:
            w.writerow([build, "image+text agent turn " + r["name"], r["prompt_tokens"], round(r["prompt_tps"],1), round(r["gen_tps"],2), round(r["ttft_s"],3), round(r.get("t_action_line_s") or 0,3), round(r.get("t_tool_call_done_s") or 0,3), round(r["total_s"],3), r["gen_tokens"], round(r["peak_mem_gb"],2), "", ""])
for build, ob in out["builds"].items():
    print("==", build)
    for r in ob["text"]: print({k: r[k] for k in ("ctx","prompt_tps","gen_tps","ttft_s","peak_mem_gb","spread_gen_tps","eff_bw_gbs_est","bw_util_pct_est")})
    for r in ob["image_turn"]: print(r["name"], r["prompt_tokens"], "ttft", r["ttft_s"], "action", r.get("t_action_line_s"), "toolcall", r.get("t_tool_call_done_s"), "total", r["total_s"], "gen", r["gen_tokens"], "tg", round(r["gen_tps"],1), "pp", round(r["prompt_tps"]), "peak", round(r["peak_mem_gb"],2), r["spread_tool_call_done_s"])
    for x in ob["demo"]: print(x["task"], x["steps"], x["task_secs"], [(s["tool"], s["secs"]) for s in x["per_step"]], x["verdict"])
