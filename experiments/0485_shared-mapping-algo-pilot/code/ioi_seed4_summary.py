"""Summary of the phase-3 seed-4 checks (stdlib only; runs on the login node).

Reads results/ioi_diag/seed4/{patch_margins.csv, probes.csv, restricted.jsonl} (ioi_diag.py seed4) and writes
results/ioi_diag/seed4/summary.md: per model, the normalised margin effect of patching (effect_norm, about 2 =
full swap, 0 = none) by layer for END and S2, the linear-probe test accuracy by layer, and the restricted IIA.

Usage: python3 ioi_seed4_summary.py <EXP>
"""
import csv
import json
import sys
from pathlib import Path


def main():
    d = Path(sys.argv[1]) / "results/ioi_diag/seed4"
    patch = list(csv.DictReader(open(d / "patch_margins.csv")))
    probe = list(csv.DictReader(open(d / "probes.csv")))
    rest = [json.loads(x) for x in open(d / "restricted.jsonl")]
    models = list(dict.fromkeys((r["state"], r["seed"]) for r in patch))
    layers = sorted({int(r["layer"]) for r in patch if int(r["layer"]) >= 0})
    out = ["# 0485 IOI phase 3: seed 4 checks", ""]
    out += ["## Unpatched models", "", "| model | task accuracy (patching pairs) | mean abs(IO-S logit) |", "|---|---|---|"]
    for st, s in models:
        r = next(r for r in patch if (r["state"], r["seed"]) == (st, s) and r["layer"] == "-1")
        out.append(f"| {st} {s} | {float(r['base_correct']):.3f} | {float(r['scale']):.3f} |")
    out.append("")
    for cand in ["pos", "alpha"]:
        for pos in ["last", "s2", "s1", "io"]:
            out += [f"## Patching effect_norm (± SE), cand {cand}, position {pos}", "",
                    "| layer | " + " | ".join(f"{st} {s}" for st, s in models) + " |", "|---|" + "---|" * len(models)]
            idx = {(r["state"], r["seed"], int(r["layer"])): r for r in patch if r["cand"] == cand and r["position"] == pos}
            for L in layers:
                out.append(f"| {L} | " + " | ".join(
                    f"{float(idx[(st, s, L)]['effect_norm']):.3f} ± {float(idx[(st, s, L)]['effect_norm_se']):.3f}"
                    for st, s in models) + " |")
            out.append("")
    for cand in ["pos", "alpha"]:
        for pos in ["last", "s2"]:
            out += [f"## Linear probe test accuracy, Z_{cand}, position {pos}", "",
                    "| layer | " + " | ".join(f"{st} {s}" for st, s in models) + " |", "|---|" + "---|" * len(models)]
            idx = {(r["state"], r["seed"], int(r["layer"])): r for r in probe if r["cand"] == cand and r["position"] == pos}
            for L in layers:
                out.append(f"| {L} | " + " | ".join(f"{float(idx[(st, s, L)]['test_acc']):.3f}" for st, s in models) + " |")
            maj = float(next(iter(idx.values()))["majority"])
            out += ["", f"majority baseline {maj:.3f}", ""]
    out += ["## IIA restricted to pairs the held-out model answers itself (default split)", "",
            "| held-out | cand | condition | n | IIA | both correct (share) | not both | both & changed | both & unchanged |",
            "|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "–" if x is None or x != x else f"{x:.3f}"
    for r in rest:
        out.append(f"| {r['heldout']} | {r['cand']} | {r['condition']} | {'' if r['adapt_examples'] is None else r['adapt_examples']} | "
                   f"{f(r['iia'])} | {f(r['iia_both_correct'])} ({f(r['own_both_correct'])}) | {f(r['iia_not_both_correct'])} | "
                   f"{f(r['iia_both_correct_changed'])} | {f(r['iia_both_correct_unchanged'])} |")
    (d / "summary.md").write_text("\n".join(out) + "\n")
    print(d / "summary.md")


if __name__ == "__main__":
    main()
