# Models I've run

Every model I've tried on my own machine. Speeds are tokens per second during generation, measured on my setup, so yours will differ.

| Date | Model | Size and quantization | Runtime | Memory used | Tokens per second | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-09 | [Qwen3 8B](https://huggingface.co/mlx-community/Qwen3-8B-4bit) | 4-bit MLX | mlx-lm | 4.726 GB | 60.8 | [your verdict] |
| 2026-10-09 | [Qwen3 14B](https://huggingface.co/lmstudio-community/Qwen3-14B-MLX-4bit) | 4-bit MLX | mlx-lm | 8.418 GB | 34.1 | [your verdict] |
| 2026-10-09 | [Gemma 3 27B](https://huggingface.co/mlx-community/gemma-3-27b-it-qat-4bit) | 4-bit MLX | mlx-lm | 16.154 GB | 17.5 | [your verdict] |
| 2026-10-09 | [Qwen3 30B-A3B (MoE)](https://huggingface.co/lmstudio-community/Qwen3-30B-A3B-MLX-4bit) | 4-bit MLX | mlx-lm | 17.234 GB | 106.1 | [your verdict] |
| 2026-10-09 | [Laguna XS 2.1](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx) | 4-bit MLX (NVFP4) | mlx-lm | 21.657 GB | 55.5 | [your verdict] |
| 2026-10-09 | [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | ternary (2-bit MLX pack) | MLX with Hadamard loader | 8.445 GB | 26.4 | [your verdict] |

I measure speed with mlx-lm's own numbers, generating 256 tokens from the same prompt for every model. Speed is generation tokens per second. Memory is peak memory reported by mlx-lm. The script is in scripts/bench.py and raw results are in results/.

Laguna XS 2.1 was run by hand with the same prompt and settings, using mlx-lm 0.32.0, because the version the script installs doesn't support it yet. Darkbloom was running during that test, so its speed may be lower than a clean run.

Ternary Bonsai 2 27B ships as PrismML's own MLX pack: ternary weights stored in a rotated (Hadamard) basis, so it needs a loader that undoes the rotation as it runs. Plain mlx-lm can't load it. I ran the original pack with that loader math on MLX, same prompt, greedy decoding. The script is scripts/bench_bonsai.py. Darkbloom was running during this test too. Raw numbers for Laguna and Bonsai are in results/2026-10-09-laguna-bonsai.json.
