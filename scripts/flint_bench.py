#!/usr/bin/env python3
"""Time Flint on any OpenAI-compatible local server (rec or Ollama). Stdlib only.
Usage:
  rec:    python3 flint_bench.py rec
  ollama: python3 flint_bench.py ollama
Prints time to first token, time to first *answer* token (after reasoning),
reasoning share, and decode tok/s. Runs 3 times and prints the median."""
import json, subprocess, sys, time, statistics, urllib.request

PROMPT = "Write a Python function that merges overlapping intervals, with 3 unit tests."
MAX_TOKENS = 1024

def target(kind):
    if kind == "rec":
        tok = subprocess.run(["rec", "auth", "token"], capture_output=True, text=True).stdout.strip()
        return "http://127.0.0.1:11500/v1/chat/completions", "flint", tok
    return "http://127.0.0.1:11434/v1/chat/completions", "flint", "ollama"

def run(url, model, key):
    body = json.dumps({"model": model, "stream": True, "max_tokens": MAX_TOKENS,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time(); t_first = t_answer = None; chunks = r_chunks = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try: ev = json.loads(data)
            except ValueError: continue
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                reasoning = d.get("reasoning_content") or d.get("reasoning")
                content = d.get("content")
                if reasoning or content:
                    now = time.time(); chunks += 1
                    if t_first is None: t_first = now
                    if reasoning: r_chunks += 1
                    if content and t_answer is None: t_answer = now
    t_end = time.time()
    n = (usage or {}).get("completion_tokens") or chunks
    decode = n / max(t_end - (t_first or t0), 1e-9)
    return {"ttft_s": round((t_first or t_end) - t0, 2),
            "first_answer_s": round((t_answer or t_end) - t0, 2),
            "reasoning_share": round(r_chunks / max(chunks, 1), 2),
            "tokens": n, "decode_tok_s": round(decode, 1)}

if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "rec"
    url, model, key = target(kind)
    run(url, model, key)  # warm-up, not counted
    rs = [run(url, model, key) for _ in range(3)]
    for r in rs: print(r)
    print(f"\n{kind}: median decode {statistics.median(r['decode_tok_s'] for r in rs)} tok/s, "
          f"median first answer token {statistics.median(r['first_answer_s'] for r in rs)} s, "
          f"reasoning share {statistics.median(r['reasoning_share'] for r in rs)}")
