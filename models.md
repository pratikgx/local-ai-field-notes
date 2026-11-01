# Models I've run

Every model I've tried on my own machine. Speeds are tokens per second during generation, measured on my setup, so yours will differ.

| Date | Model | Size and quantization | Runtime | Memory used | Tokens per second | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-09 | [Qwen3 8B](https://huggingface.co/mlx-community/Qwen3-8B-4bit) | 4-bit MLX | mlx-lm | 4.726 GB | 60.8 | Fast and light at 4.7 GB. A good everyday pick. |
| 2026-10-09 | [Qwen3 14B](https://huggingface.co/lmstudio-community/Qwen3-14B-MLX-4bit) | 4-bit MLX | mlx-lm | 8.418 GB | 34.1 | The middle ground: about twice the memory of the 8B at about half the speed. |
| 2026-10-09 | [Gemma 3 27B](https://huggingface.co/mlx-community/gemma-3-27b-it-qat-4bit) | 4-bit MLX | mlx-lm | 16.154 GB | 17.5 | Slowest here. It's dense, so all 27B parameters work on every token. |
| 2026-10-09 | [Qwen3 30B-A3B (MoE)](https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-MLX-4bit) | 4-bit MLX | mlx-lm | 17.234 GB | 106.1 | Fastest here despite being the biggest, because only about 3B parameters work per token. |
| 2026-10-09 | [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx) | 4-bit MLX (NVFP4) | mlx-lm | 21.657 GB | 55.5 | poolside's coding model. Almost as fast as the 30B for the same reason, but it needs the newest mlx-lm. |
| 2026-10-09 | [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | ternary (2-bit MLX pack) | MLX with Hadamard loader | 8.445 GB | 26.4 | The most memory-efficient here: a 27B model in 8.4 GB. It needs its own loader. |
| 2026-10-23 | [Flint 40B-A2.5B (MoE)](https://huggingface.co/RecursiveSI/flint) | 4-bit | Ollama | not measured | 115 | Read a 20K-token codebase in 11 s, then 87 tok/s. Passed 40/40 tests on the feature it added. |
| 2026-10-25 | [Nanbeige4.3-33B](https://huggingface.co/pratikgx/Nanbeige4.3-33B-MLX-4bit) | 4-bit MLX (g64), 18.6 GB | mlx-lm 0.32.2 | 19.32 GB | 14.4 | 14.4 tok/s at 512 tokens, 9.9 at 32K. It found the needle at 128K context (131K tokens, 47.75 GB peak) without running out of memory. |
| 2026-10-25 | [Qwen3.5 0.8B](https://huggingface.co/mlx-community/Qwen3.5-0.8B-4bit) | 4-bit MLX, 0.7 GB | mlx-lm | 0.97 GB | 349.8 | 1,000 Rewrites run. Lost the meaning by round 2 and shrank to 27 words. |
| 2026-10-25 | [DeepSeek-V2-Lite](https://huggingface.co/mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx) | 4-bit MLX, 8.8 GB | mlx-lm | 9.35 GB | 136.1 | 1,000 Rewrites run. Lost the meaning on round 1. |
| 2026-10-25 | [Bonsai 2 9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit) | ternary (2-bit MLX), 3.5 GB | mlx-lm | 4.35 GB | 71.4 | 1,000 Rewrites run. Lost the meaning at round 5. |
| 2026-10-25 | [Gemma 4 26B](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | 4-bit MLX, 15.4 GB | mlx-lm | 14.69 GB | 85.1 | 1,000 Rewrites run. Lost the meaning at round 3. |
| 2026-10-25 | [GLM-4.7 Flash](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) | 4-bit MLX, 16.9 GB | mlx-lm | 17.30 GB | 83.7 | 1,000 Rewrites run. Lost the meaning at round 10. |
| 2026-10-25 | [Qwen3.8 27B](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 4-bit MLX, 16.1 GB | mlx-lm | 16.15 GB | 17.7 | 1,000 Rewrites run. Lasted the longest of the models that fit in memory: 233 rounds. |
| 2026-10-25 | [Qwen3.6 35B-A3B (MoE)](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | 4-bit MLX, 20.4 GB | mlx-lm | 20.15 GB | 100.2 | 1,000 Rewrites run. Lost the meaning at round 228. |
| 2026-10-25 | [Qwen3-Next 80B (MoE)](https://huggingface.co/mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit) | 4-bit MLX, 44.9 GB | mlx-lm | 45.48 GB | 71.4 | 1,000 Rewrites run. Lost the meaning at round 222. |
| 2026-10-25 | [Qwen 3.8 Flash Next, pruned (MoE)](https://huggingface.co/neopolita/Niwaki-Qwen3.8-Flash-Next-141B-A6B-v4.0-mlx) | mixed MLX, 44.8 GB | mlx-vlm 0.7.10 | 46.88 GB | 40.3 | 1,000 Rewrites run. Lost the meaning at round 77. |
| 2026-10-25 | [GLM-5.3 Flash (MoE)](https://huggingface.co/beamster/GLM-5.3-Flash-Sushi-2bpw) | 2 bits per weight, 87.6 GB | sushi 1.3.0, experts streamed from the SSD | not measured | 5.8 | 1,000 Rewrites run. Bigger than my 64 GB, and one of two models that kept the meaning for all 1,000 rounds. |
| 2026-10-25 | [DeepSeek V4.1 Flash (MoE)](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp) | about 2.5-bit (oQ2.5e), 342.7 GB | oMLX, experts streamed from the SSD | not measured | 5.5 | 1,000 Rewrites run. More than 5x my memory, and it kept the meaning for all 1,000 rounds. |
| 2026-10-26 | [Nemotron 3.5 Bolt E4B](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit) | OptiQ mixed 4/8-bit MLX | mlx-lm | 7.28 GB | 63.0 | NVIDIA's 8B (4B effective). Still 48.2 tok/s at 32K context in 8.92 GB, and 8/8 on needle retrieval at 32K and 64K. |
| 2026-10-30 | [Mage-VL-Cowork 4B](https://huggingface.co/pratikgx/Mage-VL-Cowork-MLX-4bit) | 4-bit MLX (vision kept BF16), 3.1 GB | mlx-vlm 0.7.11 | 3.87 GB | 95.6 | Microsoft's computer-use agent. One move from a 1080p screenshot takes 3.1 s. It missed the one click test that full precision hit. |
| 2026-10-30 | [Mage-VL-Cowork 4B](https://huggingface.co/microsoft/Mage-VL-Cowork) | BF16 MLX (no quantization), 9.5 GB | mlx-vlm 0.7.11 | 9.96 GB | 30.0 | About a third the speed of 4-bit and 6.0 s per 1080p move, but it clicked the Dark Mode toggle right. |

I measure speed with mlx-lm's own numbers, generating 256 tokens from the same prompt for every model. Speed is generation tokens per second. Memory is peak memory reported by mlx-lm. The script is in scripts/bench.py and raw results are in results/.

Laguna XS 2.1 was run by hand with the same prompt and settings, using mlx-lm 0.32.0, because the version the script installs doesn't support it yet. Darkbloom was running during that test, so its speed may be lower than a clean run.

Ternary Bonsai 2 27B ships as PrismML's own MLX pack: ternary weights stored in a rotated (Hadamard) basis, so it needs a loader that undoes the rotation as it runs. Plain mlx-lm can't load it. I ran the original pack with that loader math on MLX, same prompt, greedy decoding. The script is scripts/bench_bonsai.py. Darkbloom was running during this test too. Raw numbers for Laguna and Bonsai are in results/2026-10-09-laguna-bonsai.json.

Flint ran on Ollama, not the bench script. Its speed is the median of 3 runs of one short coding prompt, timed with scripts/flint_bench.py. Memory wasn't measured. The write-up is in notes/2026-10-23-flint.md and the raw numbers are in results/2026-10-23-flint.json.

Nanbeige4.3-33B is my own 4-bit MLX build. Its speed is from a 512-token prompt with thinking on, 256 generated tokens, greedy, median of 3 runs after 1 warmup. Memory is MLX peak memory. The write-up is in notes/2026-10-25-nanbeige4.3-33b.md and the raw numbers are in results/2026-10-25-nanbeige4.3-33b.json.

The 1,000 Rewrites rows come from a different test: each model rewrote its own rewrite of the Gettysburg Address 1,000 times at temperature 0.7. Their speed is the average generation speed over all 1,000 rounds, and memory is MLX peak memory. The two SSD-streamed models ran on local servers that don't report peak memory. The write-up is in notes/2026-10-25-1000-rewrites.md and the raw numbers are in results/2026-10-25-1000-rewrites.json.

Nemotron 3.5 Bolt E4B ran on stock mlx-lm at temperature 0 with thinking on. Its speed is at 512 tokens of context, median of 3 runs. Memory is MLX peak memory. The write-up is in notes/2026-10-26-nemotron-3.5-bolt-e4b.md and the raw numbers are in results/2026-10-26-nemotron-3.5-bolt-e4b.json.

Mage-VL-Cowork is a vision model, so it ran on mlx-vlm 0.7.11 instead of the bench script. Its speeds are from a 512-token filler prompt, 256 generated tokens, greedy, median of 3 runs after 1 warmup. Memory is MLX peak memory. A background inference app was running during these tests. The write-up is in notes/2026-10-30-mage-vl-cowork.md, the scripts are in scripts/mage/ and the raw numbers are in results/2026-10-30-mage-vl-cowork.json.
