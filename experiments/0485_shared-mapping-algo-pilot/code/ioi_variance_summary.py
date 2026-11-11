"""Summary of the adaptation-variance runs (stdlib only; runs on the login node).

Reads results/ioi_diag/variance/*.jsonl (ioi_diag.py variance) and writes results/ioi_diag/variance/summary.md:
mean, SD, min and max of the held-out IIA over the repeats (repeat 0 = the original run), and, for held-out
1, 2, 5, the original folds against the replace34 folds (sources 3, 4 replaced by 6, 7).

Usage: python3 ioi_variance_summary.py <EXP>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def main():
    d = Path(sys.argv[1]) / "results/ioi_diag/variance"
    rows = [json.loads(x) for p in sorted(d.glob("*.jsonl")) for x in p.read_text().splitlines()]
    g = defaultdict(list)
    for r in rows:
        g[(r["set"], r["heldout"], r["cand"], r["condition"], r["adapt_examples"])].append(r)
    runs = list(dict.fromkeys((r["cand"], r["condition"], r["adapt_examples"]) for r in rows))
    sd = lambda v: stdev(v) if len(v) > 1 else float("nan")
    out = ["# 0485 IOI: run-to-run variance of the held-out adaptation", "",
           "Default split, trained models, Procrustes init. Only the adaptation is repeated (new minibatch seed and "
           "new random n-subset of the 4096 adaptation pairs); the shared maps are the same. Repeat 0 = original run.",
           ""]
    for tag in ["orig", "replace34"]:
        out += [f"## {tag}", "", "| held-out | cand | condition | n | repeats | repeat 0 | mean | SD | min | max |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for k in sorted(k for k in g if k[0] == tag):
            v = sorted(g[k], key=lambda r: r["repeat"])
            x = [r["iia"] for r in v]
            out.append(f"| {k[1]} | {k[2]} | {k[3]} | {k[4]} | {len(x)} | {v[0]['iia']:.3f} | {mean(x):.3f} | "
                       f"{sd(x):.3f} | {min(x):.3f} | {max(x):.3f} |")
        out.append("")
    out += ["## Original vs replace34 (held-out 1, 2, 5): mean ± SD over repeats", "",
            "| held-out | cand | condition | n | orig | replace34 | difference | difference / SE |",
            "|---|---|---|---|---|---|---|---|"]
    for h in [1, 2, 5]:
        for cand, cond, n in runs:
            a, b = g.get(("orig", h, cand, cond, n)), g.get(("replace34", h, cand, cond, n))
            if not a or not b:
                continue
            xa, xb = [r["iia"] for r in a], [r["iia"] for r in b]
            se = (sd(xa) ** 2 / len(xa) + sd(xb) ** 2 / len(xb)) ** 0.5
            diff = mean(xb) - mean(xa)
            out.append(f"| {h} | {cand} | {cond} | {n} | {mean(xa):.3f} ± {sd(xa):.3f} | {mean(xb):.3f} ± {sd(xb):.3f} | "
                       f"{diff:+.3f} | {diff / se:+.1f} |" if se > 0 else
                       f"| {h} | {cand} | {cond} | {n} | {mean(xa):.3f} | {mean(xb):.3f} | {diff:+.3f} | – |")
    (d / "summary.md").write_text("\n".join(out) + "\n")
    print(d / "summary.md")


if __name__ == "__main__":
    main()
