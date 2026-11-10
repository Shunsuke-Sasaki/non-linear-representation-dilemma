"""Compare shared_joint transfer with sources {3,4} replaced by {6,7} against the original 0485 IOI run (stdlib only).

Reads results/ioi_eval/adapt_results.csv (original; sources = the other four of seeds 1-5) and
results/ioi_diag/replace34/adapt/fold*_trained.jsonl (replacement; ioi_diag.py replace-adapt), keeps the
trained-model shared_joint rows for the held-out seeds present in both, and writes a Markdown table.

Usage: python3 ioi_replace_compare.py <EXP> [--out results/ioi_diag/replace34/compare.md]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def n_of(v):
    return int(float(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", type=Path)
    ap.add_argument("--out", default="results/ioi_diag/replace34/compare.md")
    a = ap.parse_args()
    new = {}
    sources = {}
    for p in sorted((a.exp / "results/ioi_diag/replace34/adapt").glob("fold*_trained.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["condition"] != "shared_joint":
                continue
            k = (r["split"], r["cand"], r["adapt_init"], n_of(r["adapt_examples"]), int(r["heldout"]))
            new[k] = r["iia"]
            sources[int(r["heldout"])] = r["sources"]
    old = {}
    with open(a.exp / "results/ioi_eval/adapt_results.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["state"] != "trained" or r["condition"] != "shared_joint":
                continue
            k = (r["split"], r["cand"], r["adapt_init"], n_of(r["adapt_examples"]), int(r["heldout"]))
            old[k] = float(r["iia"])
    keys = sorted(set(new) & set(old))
    held = sorted({k[4] for k in keys})
    ns = sorted({k[3] for k in keys})
    out = ["# 0485 IOI: shared_joint transfer, sources {3,4} replaced by {6,7}", "",
           "Trained models, shared_joint. old = original run (sources: the other four of seeds 1-5); "
           "new = replace34. Held-out sources in new: "
           + "; ".join(f"{h}: {sources[h]}" for h in held) + ".", ""]
    for split in ["default", "namesplit"]:
        for cand in ["pos", "alpha"]:
            for init in ["procrustes", "random_orthogonal"]:
                if not any((split, cand, init, n, h) in new for n in ns for h in held):
                    continue
                out += [f"## {split} / {cand} / {init}", "",
                        "| held-out | " + " | ".join(f"n={n} old → new" for n in ns) + " |",
                        "|---|" + "---|" * len(ns)]
                acc = defaultdict(lambda: ([], []))
                for h in held:
                    cells = []
                    for n in ns:
                        k = (split, cand, init, n, h)
                        if k in new and k in old:
                            acc[n][0].append(old[k]); acc[n][1].append(new[k])
                            cells.append(f"{old[k]:.3f} → {new[k]:.3f}")
                        else:
                            cells.append("–")
                    out.append(f"| {h} | " + " | ".join(cells) + " |")
                out.append("| mean | " + " | ".join(
                    f"{mean(acc[n][0]):.3f} → {mean(acc[n][1]):.3f}" if acc[n][0] else "–" for n in ns) + " |")
                out.append("")
    path = a.exp / a.out
    path.write_text("\n".join(out))
    print(path)


if __name__ == "__main__":
    main()
