# Setting up the Mac mini

Date: 2026-10-09

## The machine

Mac mini with an Apple M5 Pro, 64 GB of unified memory and a 1 TB drive, on macOS 27. When I'm not using it, it serves models on Darkbloom.

## What I installed

1. Apple's Command Line Tools, because macOS doesn't come with a working Python. This gave me Python 3.9.
2. mlx-lm, Apple's tool for running models with MLX. `bash scripts/bench.sh` installs it in its own environment.
3. uv, to get Python 3.12. The newest mlx-lm needs Python 3.11 or later, and Laguna and Bonsai needed the newest one.
4. Ollama and LM Studio, for chatting with models in an app. Every speed in models.md comes from MLX in the terminal, not from these apps.
5. Tailscale, for a private network between my own devices.

## First model I ran

Qwen3 8B, the 4-bit MLX build from mlx-community. The 4.3 GB download took about a minute and a half, and it wrote at 61 tokens per second. The benchmark script then ran Qwen3 14B, Gemma 3 27B and Qwen3 30B-A3B the same way. With the downloads, the whole run took about 15 minutes.

## What surprised me

The biggest model was the fastest. Qwen3 30B-A3B ran at 106 tokens per second because only about 3B of its parameters work on each token. Laguna XS 2.1 works the same way and ran at 56.

Ternary Bonsai 2 27B stores nearly every weight as -1, 0 or 1. It ran at 26 tokens per second in 8.4 GB, faster than Gemma 3 27B in about half the memory.

## What broke

- **No Python.** The first run failed straight away. macOS asked to install the Command Line Tools, and I had to click through that installer myself.
- **Too old for new models.** The Command Line Tools give you Python 3.9, which caps mlx-lm at version 0.29.1. That version can't run Laguna. I installed uv, made a separate Python 3.12 environment, and Laguna ran on mlx-lm 0.32.0.
- **Bonsai needs its own loader.** Its weights are stored in a rotated form. Plain mlx-lm refuses to load it, and PrismML warns that loaders which skip the rotation give garbage with no error. It runs correctly with a loader that undoes the rotation as it goes. That's scripts/bench_bonsai.py.
- **A harmless warning.** Laguna prints a tokenizer warning when it loads. Its answers still came out clean.
- **Pause other GPU work first.** Darkbloom uses the same GPU and memory, so I paused it for the main benchmark. It was running but idle during the Laguna and Bonsai runs.
