# 1,000 Rewrites Later scripts

`analyze.py` computes the drift metrics for rewrite-loop logs: similarity to the original and to the previous round, word count, unique-word share, repeated trigrams, and the collapse round.

It reads one JSON Lines log per model (one row per round, with the round's text), so you can point it at your own runs:

```bash
pip install sentence-transformers matplotlib
python analyze.py "runs/*.jsonl" --out analysis
```

Without sentence-transformers, `--embed tfidf` uses a simpler TF-IDF similarity (use a lower threshold, around 0.5).

`seed_gettysburg.txt` is the starting text. The write-up is in `notes/2026-10-25-1000-rewrites.md`, and every round from my runs is in the Hugging Face dataset [pratikgx/1000-rewrites-later](https://huggingface.co/datasets/pratikgx/1000-rewrites-later).
