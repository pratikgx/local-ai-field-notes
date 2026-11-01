# Launch bench speed script

`bench_speed.py` measures prefill speed, decode speed, time to first token and MLX peak memory for any mlx-lm model, using the fixed settings in `settings.json`: greedy decoding, prompts of 512, 2,048, 8,192 and 16,000 tokens, 256 generated tokens with EOS ignored, prefill step 2048, 1 warmup and then 3 timed runs (it saves the median, min and max).

The prompt is the BOS token plus the first N tokens of *Pride and Prejudice*. Put the plain text from [Project Gutenberg](https://www.gutenberg.org/ebooks/1342) at `data/pride_and_prejudice.txt`, one folder up from this one, and make a `runs/` folder next to it. Results are written to `runs/speed_<tag>.json`.

```bash
pip install mlx-lm
mkdir -p ../data ../runs
python bench_speed.py udgamlabs/Laghu-MLX bf16
python bench_speed.py pratikgx/Laghu-MLX-4bit q4
```

The script caps prompts at 16,384 tokens minus the generated tokens (`maxctx`), which was Laghu's context limit. Raise it for models with a longer context.

The quality side of this harness (GSM8K, IFEval, ARC-Easy, HellaSwag) is in the Hugging Face dataset [pratikgx/launch-bench-results](https://huggingface.co/datasets/pratikgx/launch-bench-results).
