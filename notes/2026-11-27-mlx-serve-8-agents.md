# mlx-serve 26.11.4, and 8 coding agents on a 64GB M5 Pro Mac mini

Date: 27 November 2026

I ran [mlx-serve](https://github.com/ddalcu/mlx-serve) 26.11.4, David Dalcu's ([@ddalcu](https://x.com/ddalcu)) native Zig server (it bundles mlx 0.32.5), on my 64GB M5 Pro Mac mini. Two parts: a speed comparison against plain mlx-lm 0.32.2 on the same Qwen3.8-27B 4-bit and 8-bit files, then a 2-hour run of 8 pi coding agents on one mlx-serve instance serving Inkling 2 Mini. The agent run was inspired by [his post](https://x.com/ddalcu/status/2126331169893679236).

Hardware: Apple M5 Pro, 64GB unified memory, Mac mini, macOS 27.2.1 (build 26B131), AC power. A background app was left running at Pratik's request and was not paused or stopped. No TensorFold in this packet.

## Headline

With speculation off, mlx-serve and mlx-lm stay within 3 to 11% on Qwen3.8-27B prose decode. With mlx-serve's shipping defaults (native MTP and prompt-lookup decoding on), prose is 1.56x to 2.08x the mlx-lm baseline. On an "echo and edit a file" prompt, MTP alone reached 121.25 tok/s on the 4-bit build and 88.57 tok/s on the 8-bit build.

The 8-agent run generated 336,674 tokens in 2 hours, 46.6 tok/s average, 110.9 tok/s peak over a 2-minute window, 665 requests ok, 0 failed, 35.1 GiB peak, and 8/8 apps passing a basic run check.

## Phase 1: Qwen3.8-27B, mlx-serve vs mlx-lm

Same weight files for both engines: [EigenLabs/Qwen3.8-27B-4bit-mtp](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit-mtp) (16.3 GB) and [bluehawana/Qwen3.8-27B-8bit-MTP-MLX](https://huggingface.co/bluehawana/Qwen3.8-27B-8bit-MTP-MLX) (30.0 GB). mlx-lm drops the MTP tensors at load, so the trunk is the same bytes. Thinking off, temperature 0, 256 new tokens, one discarded warmup, median of 3 timed runs, a unique prompt each run so the cache cannot help. mlx-serve `plain` is `--no-mtp --no-pld`. `default` is shipping defaults. `mtp_only` is `--no-pld`.

`brew install mlx-serve` failed on this machine (the tap cloned, then brew reported "No available formula"), so I installed the official release tarball, whose sha256 matches the formula.

### Prose decode, speculation off

| build | prompt | mlx-lm | mlx-serve (off) | diff |
|---|---:|---:|---:|---:|
| 4-bit | 512 | 16.80 | 18.51 | +10% |
| 4-bit | 4K | 17.23 | 18.09 | +5% |
| 4-bit | 32K | 14.80 | 15.19 | +3% |
| 8-bit | 512 | 9.66 | 9.97 | +3% |
| 8-bit | 4K | 9.01 | 9.98 | +11% |
| 8-bit | 32K | 9.04 | 9.35 | +3% |

### mlx-serve defaults (MTP + PLD on)

| build | prompt | mlx-lm | mlx-serve defaults | speedup |
|---|---:|---:|---:|---:|
| 4-bit | 512 | 16.80 | 26.29 | 1.56x |
| 4-bit | 4K | 17.23 | 30.07 | 1.75x |
| 8-bit | 512 | 9.66 | 20.05 | 2.08x |
| 8-bit | 4K | 9.01 | 18.21 | 2.02x |

32K was not run with defaults, to save time. Prefill and time to first token are a little slower on mlx-serve with speculation off (4-bit 512: 1.26 s vs 1.13 s). Charts: [decode speed](img/2026-11-27-mlx-serve-decode-speed.png).

### Code prompts

2 runs, 320 new tokens. Speedups are vs mlx-serve with speculation off. On this Mac, MTP alone beat the shipping default in all four cells.

| build | prompt | mlx-lm | mlx-serve off | MTP only | defaults |
|---|---|---:|---:|---:|---:|
| 4-bit | write code from scratch | 17.74 | 17.71 | **49.86** (2.8x) | 35.67 |
| 4-bit | echo + edit a file | 17.63 | 18.69 | **121.25** (6.5x) | 105.00 |
| 8-bit | write code from scratch | 9.74 | 10.08 | **35.72** (3.5x) | 30.30 |
| 8-bit | echo + edit a file | 9.71 | 9.40 | **88.57** (9.4x) | 73.23 |

Chart: [code speculation](img/2026-11-27-mlx-serve-code-spec-decode.png). The file-echo prompt produced byte-identical output across every engine and mode. The write-from-scratch prompt did not; the split is from small numeric differences between the two engines, not from MTP changing the text.

Plain decode is already close to the 307 GB/s memory-bandwidth limit (about 79 to 88%), which is why the two engines are close with speculation off. Peak `phys_footprint_peak` for the 4-bit 32K sweep was 28.5 GiB on mlx-lm and 23.1 GiB on mlx-serve with speculation off. mlx-lm keeps a prompt cache of past requests, so most of that gap is cache policy.

## Phase 2: 8 agents, Inkling 2 Mini

One mlx-serve instance, `--max-concurrent 8 --metrics`, default speculation, serving official [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) weights (Apache 2.0). Those weights are mixed precision: NVFP4 experts in 20 layers, MXFP8 in 4 layers, BF16/F32 elsewhere. They are not a true 8-bit build. I skipped a true 8-bit Inkling run because the Mac had only about 27 to 28 GB free, and the official weights were already on disk. The LM Studio "MLX-8bit" pack still keeps NVFP4 experts, so it would not have been a true 8-bit comparison either.

8 pi 0.73.1 coding agents, each sandboxed to its own folder, each given a fixed SPEC.md (snake, expense tracker, kanban, pomodoro, markdown-to-HTML, todo API, unit converter, typing test). 120 minute hard window, 30 minute session cap, then a continue/improve session. The first launch failed because mlx-serve answered `/v1/models` before the model had loaded. I fixed the id, restarted only the agents, and then started the clock.

| metric | value |
|---|---|
| tokens generated | 336,674 |
| prompt tokens processed | 6,801,038 (3,004,979 from the prefix cache) |
| average aggregate output | 46.6 tok/s |
| peak 2-minute aggregate | 110.9 tok/s |
| median 2-minute aggregate | 41.7 tok/s |
| requests | 665 ok, 0 failed |
| server peak memory | 35.1 GiB |
| apps passing the basic check | 8 / 8 |
| output | 51 files, 6,179 lines |

Each agent averaged 5.2 to 7.8 tok/s while its requests were in flight, including queueing and prefill next to the other seven. Charts: [aggregate tok/s](img/2026-11-27-mlx-serve-8-agents-aggregate.png), [per agent](img/2026-11-27-mlx-serve-8-agents-per-agent.png).

The basic check was written before the run. For browser apps it means the page loads, JS parses, required element ids are present, and the initial state matches the spec. For the Python apps it means the agent's unit tests pass, plus a hidden markdown check and a CRUD check on the todo API. That is not full QA. These are AI-built demos from a 2-hour unattended run. The finished apps are in [pratikgx/mlx-serve-8-agents](https://github.com/pratikgx/mlx-serve-8-agents).

## Caveats

- The background app was running the whole time, and its load jumped around. One mlx-lm 4-bit sweep under heavy load (10.3 tok/s at 512) was discarded and redone. Some cells still have over 10% spread, including mlx-serve 4-bit plain at 32K (26%) and 4-bit defaults at 4K (23%).
- mlx-serve bundles mlx 0.32.5; the mlx-lm baseline used mlx 0.32.4.
- Code prompts used only 2 runs each. Defaults were not run at 32K.
- Inkling in phase 2 is official mixed NVFP4/MXFP8, not 8-bit.

## Files

- img/2026-11-27-mlx-serve-decode-speed.png, img/2026-11-27-mlx-serve-code-spec-decode.png, img/2026-11-27-mlx-serve-8-agents-aggregate.png, img/2026-11-27-mlx-serve-8-agents-per-agent.png
- The public dataset has the results CSVs, all eight charts, raw phase 1 JSON, scripts, and screenshots of the finished apps.

## License and credit

Measurements: CC BY 4.0, attribution Pratik Gandhi. Credit to David Dalcu for mlx-serve, to Thinking Machines for Inkling 2 Mini, and to the Qwen3.8-27B MLX pack authors. No model weights are redistributed in the dataset.

## Data

- Hugging Face benchmark data: [pratikgx/mlx-serve-26.11.4-m5-pro-mac-mini-bench](https://huggingface.co/datasets/pratikgx/mlx-serve-26.11.4-m5-pro-mac-mini-bench)
- Apps: [pratikgx/mlx-serve-8-agents](https://github.com/pratikgx/mlx-serve-8-agents)
