# Mistral Large 4 (1T) on a 64 GB Mac mini

Date: 2026-10-29

Mistral Large 4, nicknamed "Le Chonk", is a 1 trillion parameter open-weight model with 52B active per token (mixture of experts). The very aggressive IQ1_S quant I used is still 233 GB, way more than my 64 GB. So the question was whether it would run at all by streaming the experts off the SSD. It does, slowly. All credit for the model goes to [Mistral AI](https://huggingface.co/mistralai), and the GGUF quants are by [unsloth](https://huggingface.co/unsloth/Mistral-Large-4-1T-A52B-GGUF).

## The setup

Mac mini, Apple M5 Pro, 64 GB, macOS 27.0.1, plugged in. A background inference app was stopped for the timed runs.

- **Model:** `unsloth/Mistral-Large-4-1T-A52B-GGUF`, quant `UD-IQ1_S`, 6 shards, 233.0 GB in total
- **Engine:** llama.cpp built from master at commit `3c809f4` (v0.8.0-dev), which includes the Le Chonk support from PR #31376. Metal backend, mmap on (the default).
- **How it fits:** with `--cpu-moe`, the routed experts stay on the CPU side and get paged in from the SSD through mmap, while the dense and attention weights run on the GPU.

## How I measured

```
llama-completion -m Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf \
  -f p512.txt -c 2048 -n 128 --ignore-eos --temp 0 --seed 42 -no-cnv \
  -ngl 99 --cpu-moe -fa on
```

- A 441-token prompt, 128 generated tokens, greedy, seed 42
- 3 timed runs at the default thread count (6), then 1 run with `-t 12`
- Speeds come from llama.cpp's own timings. Time to first token is the prompt eval time plus one decode step.
- Memory comes from `/usr/bin/time -l`

## Results

| Run | Threads | Prefill tok/s | Decode tok/s | Time to first token | Load | Peak RSS | Peak footprint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run1 | 6 | 3.02 | 0.36 | 148.9 s | 45.0 s | 58.3 GB | 2.05 GB |
| run2 | 6 | 2.99 | 0.35 | 150.2 s | 55.0 s | 56.8 GB | 2.02 GB |
| run3 | 6 | 3.00 | 0.35 | 150.1 s | 54.3 s | 54.2 GB | 2.05 GB |
| **median, 6 threads** | 6 | **3.00** | **0.35** | **150.1 s** | **54.3 s** | **56.8 GB** | **2.05 GB** |
| run_t12 (single run) | 12 | 2.98 | 0.49 | 149.9 s | 52.5 s | 53.6 GB | 2.04 GB |

- **Memory:** the 54 to 58 GB peak RSS is mostly memory-mapped model pages that macOS can drop and reload. The actual peak footprint was about 2 GB. Each timed run had roughly 50 million page faults, which is the SSD streaming at work.
- **Full GPU offload doesn't fit:** `-ngl 99` without `--cpu-moe` failed with a Metal out-of-memory error.
- **Output:** coherent. It gave a correct 2-sentence summary of the prompt document, then repeated itself because of `--ignore-eos`.
- **Threads:** 12 threads got decode to 0.49 tok/s, but that's a single run.

## Caveats

- Speed only, no quality suite. IQ1_S is a very aggressive quant, so don't read anything about quality into this.
- Short prompt (441 tokens) and short context (2,048) only.
- llama.cpp's Le Chonk support was brand new, so speeds may change. llama.cpp also suggested `--load-mode none` might beat mmap with CPU tensor overrides. I didn't test that.
- SSD speed matters a lot here. Your numbers will depend on your drive.

## Files

- Results: `results/2026-10-29-mistral-large-4-1t.json` and `results/2026-10-29-mistral-large-4-1t.csv` (including a short 16-token probe and the failed full-GPU probe)
- Hugging Face dataset with the raw llama.cpp output for every run: [pratikgx/mistral-large-4-1t-m5-pro-mac-mini-bench](https://huggingface.co/datasets/pratikgx/mistral-large-4-1t-m5-pro-mac-mini-bench)
