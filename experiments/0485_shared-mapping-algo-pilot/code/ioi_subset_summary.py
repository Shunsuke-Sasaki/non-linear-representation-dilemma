"""Supplementary IOI summary restricted to a subset of held-out seeds (stdlib only; runs on the login node).

Reads results/ioi_eval/adapt_results.csv written by ioi_eval.py and writes a Markdown table of held-out IIA
averaged over the chosen held-out seeds only. It does NOT retrain anything: in every fold the shared map was
still trained on the other four seeds, so excluded seeds can remain among the sources.

Usage: python3 ioi_subset_summary.py <results/ioi_eval> --heldout 1 2 5 [--out summary_heldout125.md]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

CONDITIONS = ["no_shared", "single_model", "shared_joint"]
INITS = ["procrustes", "random_orthogonal"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x):
    return "–" if x is None else f"{x:.3f}"


def avg(rows, col):
    vals = [float(r[col]) for r in rows if r[col] != ""]
    return mean(vals) if vals else None


def build(rows, heldout, all_seeds):
    rows = [r for r in rows if int(r["heldout"]) in heldout]
    groups = defaultdict(list)
    for r in rows:
        n = "" if r["adapt_examples"] == "" else int(float(r["adapt_examples"]))
        groups[(r["state"], r["split"], r["cand"], r["condition"], r["adapt_init"], n)].append(r)
    ns = sorted({k[5] for k in groups if k[5] != ""})
    hs = ", ".join(map(str, sorted(heldout)))
    out = [
        f"# 0485 IOI supplementary summary: held-out seeds {hs} only",
        "",
        f"Source: `adapt_results.csv`. Mean over the held-out seeds {hs} (one fold each). Excluded as targets: "
        f"{', '.join(map(str, sorted(set(all_seeds) - set(heldout))))}.",
        "The maps were not retrained: each fold's shared / single-model map still comes from the other four seeds, "
        "so excluded seeds remain among the sources. `summary.md` (all five seeds) is unchanged.",
        "",
    ]
    for state in ["trained", "random"]:
        out += [f"## {state}: held-out IIA", ""]
        out.append("| split | cand | init | condition | " + " | ".join(f"n={n}" for n in ns) + " | self |")
        out.append("|---|---|---|---|" + "---|" * (len(ns) + 1))
        for split in ["default", "namesplit"]:
            for cand in ["pos", "alpha"]:
                self_iia = avg(groups.get((state, split, cand, "individual_self", "", ""), []), "iia")
                for init in INITS:
                    for cond in CONDITIONS:
                        vals = [avg(groups.get((state, split, cand, cond, init, n), []), "iia") for n in ns]
                        if all(v is None for v in vals):
                            continue
                        out.append(f"| {split} | {cand} | {init} | {cond} | " + " | ".join(map(fmt, vals))
                                   + f" | {fmt(self_iia)} |")
        out.append("")
    nmax = ns[-1]
    out += [f"## IIA split by whether the label changes (n={nmax}, procrustes)", ""]
    out.append("| state | split | cand | condition | label changed | label unchanged |")
    out.append("|---|---|---|---|---|---|")
    for state in ["trained", "random"]:
        for split in ["default", "namesplit"]:
            for cand in ["pos", "alpha"]:
                for cond in CONDITIONS:
                    g = groups.get((state, split, cand, cond, "procrustes", nmax), [])
                    if g:
                        out.append(f"| {state} | {split} | {cand} | {cond} | {fmt(avg(g, 'iia_label_changed'))} | "
                                   f"{fmt(avg(g, 'iia_label_unchanged'))} |")
    out.append("")
    out += [f"## Per held-out seed: trained, procrustes, shared_joint (and self)", ""]
    out.append("| split | cand | seed | self | " + " | ".join(f"n={n}" for n in ns) + " |")
    out.append("|---|---|---|---|" + "---|" * len(ns))
    for split in ["default", "namesplit"]:
        for cand in ["pos", "alpha"]:
            for s in sorted(heldout):
                one = lambda c, i, n: avg([r for r in groups.get(("trained", split, cand, c, i, n), [])
                                           if int(r["heldout"]) == s], "iia")
                out.append(f"| {split} | {cand} | {s} | {fmt(one('individual_self', '', ''))} | "
                           + " | ".join(fmt(one("shared_joint", "procrustes", n)) for n in ns) + " |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--heldout", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.results_dir / "adapt_results.csv")
    all_seeds = sorted({int(r["heldout"]) for r in rows})
    text = build(rows, set(a.heldout), all_seeds)
    out = a.results_dir / (a.out or f"summary_heldout{''.join(map(str, sorted(a.heldout)))}.md")
    out.write_text(text)
    print(out)


if __name__ == "__main__":
    main()
