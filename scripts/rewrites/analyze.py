#!/usr/bin/env python3
"""
Analyze 1,000 Rewrites Later logs.

Per model log (runs/*.jsonl), per round:
  sim_to_original  cosine similarity of the round's text to the seed (embeddings)
  sim_to_prev      cosine similarity to the previous round (size of each step)
  words            word count
  unique_ratio     unique words / words (falls as text gets repetitive)
  rep_trigram      share of repeated word trigrams (rises when a model loops)
  same_meaning     sim_to_original >= --threshold  (the "still the same meaning?" flag)

Writes into --out (default analysis/):
  metrics.csv, leaderboard.csv, leaderboard.md, checkpoints.md (text at 0/10/50/100/250/500/1000),
  progress.md (latest checkpoint per model; handy for posting partial results),
  similarity_vs_round.png (hero chart, all models), words_vs_round.png, unique_ratio_vs_round.png

Unequal round counts are expected: the frontier models stream from the SSD and may sit at round 50
while the others are at 1,000. Each run's target comes from <label>.status.json (run_rewrites.py),
else models.conf, else the log's meta row. A run that has not reached its target and has not
collapsed is "running" and is never called "held the meaning"; it is listed after the finished runs.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 by default. Without
sentence-transformers (or with --embed tfidf) a TF-IDF cosine fallback runs instead.

  python analyze.py "runs/*.jsonl"
  python analyze.py "runs/*.jsonl" --embed tfidf
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter

try:
    import lineup as L
except ImportError:  # analyze.py copied somewhere on its own
    L = None

CHECKPOINTS = [0, 10, 50, 100, 250, 500, 1000]
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def words_of(text):
    return WORD_RE.findall(text.lower())


def load_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    rows.sort(key=lambda r: r["round"])
    return rows


class STEmbedder:
    def __init__(self, name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)
        self.kind = f"sentence-transformers:{name}"

    def encode(self, texts):
        import numpy as np
        return np.asarray(self.m.encode(texts, batch_size=64, normalize_embeddings=True,
                                        show_progress_bar=False))


class TfidfEmbedder:
    """Dependency-free fallback. Coarser than real embeddings, so use a lower threshold (~0.5)."""
    kind = "tfidf"

    def encode(self, texts):
        import numpy as np
        docs = [Counter(words_of(t)) for t in texts]
        df = Counter()
        for d in docs:
            df.update(d.keys())
        vocab = {w: i for i, w in enumerate(df)}
        n = len(docs)
        mat = np.zeros((n, max(1, len(vocab))), dtype="float32")
        for i, d in enumerate(docs):
            for w, c in d.items():
                mat[i, vocab[w]] = (1 + math.log(c)) * math.log((1 + n) / (1 + df[w]) + 1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms


def make_embedder(kind, name):
    if kind == "st":
        try:
            return STEmbedder(name)
        except Exception as e:
            print(f"[warn] sentence-transformers unavailable ({e.__class__.__name__}); using TF-IDF fallback")
    return TfidfEmbedder()


def text_stats(text):
    w = words_of(text)
    n = len(w)
    uniq = len(set(w)) / n if n else 0.0
    tri = [tuple(w[i:i + 3]) for i in range(max(0, n - 2))]
    rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
    return n, uniq, rep


def collapse_round(series, threshold, seed_words):
    """First round where the model has clearly lost the original, sustained for 3 rounds."""
    streak = 0
    for row in series[1:]:
        lost = (row["sim_to_original"] < threshold or row["rep_trigram"] > 0.5
                or row["words"] < 0.3 * seed_words or row["status"] != "ok")
        streak = streak + 1 if lost else 0
        if streak >= 3:
            return row["round"] - 2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help='e.g. "runs/*.jsonl"')
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--embed", choices=["st", "tfidf"], default="st")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=None,
                    help="same-meaning cutoff (default 0.75 MiniLM, 0.5 TF-IDF); calibrate on the pilot")
    args = ap.parse_args()

    paths = []
    for p in args.logs:
        paths.extend(sorted(glob.glob(p)) or [p])
    paths = [p for p in paths if p.endswith(".jsonl")]
    os.makedirs(args.out, exist_ok=True)
    emb = make_embedder(args.embed, args.embed_model)
    thr = args.threshold if args.threshold is not None else (0.5 if emb.kind == "tfidf" else 0.75)
    print(f"[embed] {emb.kind}  threshold={thr}")

    all_series = {}
    for path in paths:
        rows = load_log(path)
        if len(rows) < 2:
            print(f"[skip] {path}: fewer than 2 rounds")
            continue
        label = rows[0].get("label") or os.path.basename(path)[:-6]
        texts = [r.get("text") or "" for r in rows]
        vecs = emb.encode(texts)
        sims0 = vecs @ vecs[0]
        series = []
        for i, r in enumerate(rows):
            n, uniq, rep = text_stats(texts[i])
            series.append({
                "model": label, "round": r["round"],
                "sim_to_original": round(float(sims0[i]), 4),
                "sim_to_prev": round(float(vecs[i] @ vecs[i - 1]), 4) if i else 1.0,
                "words": n, "unique_ratio": round(uniq, 4), "rep_trigram": round(rep, 4),
                "same_meaning": bool(sims0[i] >= thr),
                "gen_tps": r.get("gen_tps"), "wall_s": r.get("wall_s"),
                "status": r.get("status", "ok"), "preamble": r.get("preamble", False),
                "text": texts[i],
            })
        last = rows[-1]["round"]
        runs_dir = os.path.dirname(os.path.abspath(path))
        target = L.target_for(label, runs_dir, rows[0], last) if L else max(rows[0].get("rounds") or last, last)
        st = L.read_status(runs_dir, label) if L else None
        for x in series:
            x["_target"] = target
            x["_state"] = (st or {}).get("state")
        all_series[label] = series
        print(f"[ok] {label}: {len(series) - 1} rounds" + ("" if last >= target else f" (of {target}, still running)"))
    if not all_series:
        raise SystemExit("no usable logs")

    fields = ["model", "round", "sim_to_original", "sim_to_prev", "words", "unique_ratio",
              "rep_trigram", "same_meaning", "gen_tps", "wall_s", "status", "preamble"]
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in all_series.values():
            w.writerows(s)

    board = []
    for label, s in all_series.items():
        last = s[-1]
        cr = collapse_round(s, thr, s[0]["words"])
        tps = [x["gen_tps"] for x in s[1:] if x["gen_tps"]]
        walls = [x["wall_s"] for x in s[1:] if x["wall_s"]]
        target = s[0]["_target"]
        complete = last["round"] >= target or last["status"] != "ok"
        if cr is not None:
            status = "collapsed"
        elif complete:
            status = "held"
        else:
            status = "running"
        board.append({
            "model": label, "rounds": last["round"], "target": target, "status": status,
            "collapse_round": cr if cr is not None else ("never" if complete else "not yet"),
            "final_similarity": last["sim_to_original"],
            "seed_words": s[0]["words"], "final_words": last["words"],
            "final_unique_ratio": last["unique_ratio"],
            "avg_tok_s": round(sum(tps) / len(tps), 1) if tps else "",
            "hours": round(sum(walls) / 3600, 2) if walls else "",
        })
    rank = {"collapsed": 0, "held": 1, "running": 2}
    board.sort(key=lambda b: (rank[b["status"]],
                              b["collapse_round"] if b["status"] == "collapsed" else 0,
                              b["final_similarity"] if b["status"] != "running" else -b["rounds"]))
    with open(os.path.join(args.out, "leaderboard.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(board[0].keys()))
        w.writeheader()
        w.writerows(board)
    with open(os.path.join(args.out, "leaderboard.md"), "w") as f:
        f.write("| # | Model | Rounds | Collapses at round | Similarity at last round | Words (seed -> last) | tok/s | Hours |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, b in enumerate(board, 1):
            rounds = f"{b['rounds']:,}" + ("" if b["rounds"] >= b["target"] else f" of {b['target']:,} (running)")
            f.write(f"| {i} | {b['model']} | {rounds} | {b['collapse_round']} | {b['final_similarity']} | "
                    f"{b['seed_words']} -> {b['final_words']} | {b['avg_tok_s']} | {b['hours']} |\n")

    with open(os.path.join(args.out, "progress.md"), "w") as f:
        f.write("| Model | Rounds | Latest checkpoint | Similarity there | Status |\n|---|---|---|---|---|\n")
        for b in board:
            s = all_series[b["model"]]
            by_round = {x["round"]: x for x in s}
            cps = [c for c in CHECKPOINTS if c in by_round and c > 0]
            c = cps[-1] if cps else None
            f.write(f"| {b['model']} | {b['rounds']:,} / {b['target']:,} | {c if c else '-'} | "
                    f"{by_round[c]['sim_to_original'] if c else '-'} | {b['status']} |\n")

    with open(os.path.join(args.out, "checkpoints.md"), "w") as f:
        for label, s in all_series.items():
            by_round = {x["round"]: x for x in s}
            tgt = s[0]["_target"]
            f.write(f"# {label}" + ("" if s[-1]["round"] >= tgt else f"  (still running: round {s[-1]['round']} of {tgt})") + "\n\n")
            for c in CHECKPOINTS:
                if c in by_round:
                    x = by_round[c]
                    f.write(f"## Round {c}  (similarity {x['sim_to_original']}, {x['words']} words)\n\n"
                            f"{x['text']}\n\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def chart(key, ylabel, fname, hline=None):
            fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)  # 16:9 for video
            ax.set_prop_cycle(color=plt.cm.tab20.colors)          # 11 models + hot runs: no repeats
            for label, s in all_series.items():
                running = s[-1]["round"] < s[0]["_target"] and s[-1]["status"] == "ok"
                name = label + (f" (round {s[-1]['round']}, running)" if running else "")
                line, = ax.plot([x["round"] for x in s], [x[key] for x in s], label=name, linewidth=2)
                if running:
                    ax.plot([s[-1]["round"]], [s[-1][key]], "o", color=line.get_color(), markersize=6)
            if hline is not None:
                ax.axhline(hline, linestyle="--", linewidth=1, color="gray")
                ax.annotate("same-meaning line", xy=(0, hline), xytext=(4, 4),
                            textcoords="offset points", color="gray")
            ax.set_xlabel("Rewrite round")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, fname))
            plt.close(fig)

        chart("sim_to_original", "Similarity to the original text", "similarity_vs_round.png", hline=thr)
        chart("words", "Words", "words_vs_round.png")
        chart("unique_ratio", "Unique-word ratio", "unique_ratio_vs_round.png")
    except ImportError:
        print("[warn] matplotlib not installed; skipped charts (pip install matplotlib)")

    print(open(os.path.join(args.out, "leaderboard.md")).read())
    print(f"[done] wrote {args.out}/")


if __name__ == "__main__":
    main()
