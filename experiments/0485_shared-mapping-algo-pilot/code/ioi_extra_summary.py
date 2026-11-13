#!/usr/bin/env python3
"""IOI phase 7, extra held-out models (seeds 6-9), stdlib only.
Usage: python3 ioi_extra_summary.py <EXP>

Analysis fixed before the xadapt results were seen:
  unit       one held-out model. Its value = mean over the joint maps (folds) applied to it of the mean over
             adaptation repeats. Seeds 1-5: the one fold that held it out (results/ioi_ioname_stable/adapt);
             seeds 6-9: all 5 folds (results/ioi_ioname_stable/extra/adapt).
  groups     primary: seeds 6-9 (never used for any map, selection or earlier transfer table);
             secondary: 1,2,5,6-9 (seeds 3 and 4 left out as in the earlier tables) and all 9.
  metrics    4-way IIA and vocabulary argmax of shared_joint (n = 0, 256, 1024, 4096) and no_shared (n = 4096).
  contrasts  paired over models: C_pos - C_name (shared, n=4096) and shared - no_shared (n=4096), per split.
  interval   mean ± t(0.975, k-1) * SD / sqrt(k) over the k models of the group.
  passed     the same with only the joint maps that fit every source (2-way source test IIA >= 0.95).
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SIZES = [0, 256, 1024, 4096]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
GROUPS = {"6-9 (primary)": [6, 7, 8, 9], "1,2,5,6-9": [1, 2, 5, 6, 7, 8, 9], "all 9": list(range(1, 10))}
CANDS, SPLITS = ["pos", "name"], ["default", "namesplit"]


def ci(v):
    if len(v) < 2:
        return f"{v[0]:.3f}" if v else ""
    h = T975[len(v) - 1] * st.stdev(v) / len(v) ** 0.5
    return f"{st.mean(v):.3f} ± {h:.3f}"


def main():
    root = Path(sys.argv[1]) / "results" / "ioi_ioname_stable"
    rows = [json.loads(x) for f in sorted((root / "adapt").glob("*.jsonl")) for x in f.read_text().splitlines()]
    rows += [json.loads(x) for f in sorted((root / "extra" / "adapt").glob("*.jsonl")) for x in f.read_text().splitlines()]
    reps = defaultdict(list)  # (passed_only, split, cand, cond, n, target, fold, metric) -> values over repeats
    for r in rows:
        if r["condition"] == "individual_self":
            continue
        tgt = r.get("target", r["heldout"])
        for m in ("iia4", "iia_vocab"):
            v = r[f"{r['trained_on']}_{m}"]
            reps[(False, r["split"], r["trained_on"], r["condition"], r["adapt_examples"], tgt, r["fold"], m)].append(v)
            if r.get("joint_passed"):
                reps[(True, r["split"], r["trained_on"], r["condition"], r["adapt_examples"], tgt, r["fold"], m)].append(v)
    per = defaultdict(lambda: defaultdict(list))  # (passed, split, cand, cond, n, metric) -> target -> fold means
    for (p, split, cand, cond, n, tgt, fold, m), v in reps.items():
        per[(p, split, cand, cond, n, m)][tgt].append(st.mean(v))

    def unit(p, split, cand, cond, n, m="iia4"):
        return {t: st.mean(v) for t, v in per[(p, split, cand, cond, n, m)].items()}

    out = ["# IOI phase 7: extra held-out models (seeds 6-9)", "", "Analysis" + __doc__.split("Analysis", 1)[1], ""]
    for passed in (False, True):
        out += [f"## {'Only joint maps that fit every source' if passed else 'All joint maps'}", ""]
        for split in SPLITS:
            out += [f"### {split}: mean ± 95% CI over held-out models (4-way IIA; vocabulary in brackets)", "",
                    "| group | cand | no_shared n=4096 | shared n=0 | n=256 | n=1024 | n=4096 | n=4096 vocab |",
                    "|---|---|---|---|---|---|---|---|"]
            for g, ts in GROUPS.items():
                for cand in CANDS:
                    def col(cond, n, m="iia4"):
                        u = unit(passed, split, cand, cond, n, m)
                        v = [u[t] for t in ts if t in u]
                        return ci(v) + ("" if len(v) == len(ts) else f" (k={len(v)})")
                    out.append(f"| {g} | {cand} | {col('no_shared', 4096)} | " + " | ".join(
                        col("shared_joint", n) for n in SIZES) + f" | {col('shared_joint', 4096, 'iia_vocab')} |")
            out += ["", "Paired contrasts, n=4096 (mean ± 95% CI over models; k = models with both values):", "",
                    "| group | pos - name (4-way) | pos - name (vocab) | pos: shared - no_shared | name: shared - no_shared |",
                    "|---|---|---|---|---|"]
            for g, ts in GROUPS.items():
                def diff(a, b):
                    v = [a[t] - b[t] for t in ts if t in a and t in b]
                    return ci(v) + f" (k={len(v)})" if v else ""
                out.append(f"| {g} | {diff(unit(passed, split, 'pos', 'shared_joint', 4096), unit(passed, split, 'name', 'shared_joint', 4096))} | "
                           f"{diff(unit(passed, split, 'pos', 'shared_joint', 4096, 'iia_vocab'), unit(passed, split, 'name', 'shared_joint', 4096, 'iia_vocab'))} | "
                           f"{diff(unit(passed, split, 'pos', 'shared_joint', 4096), unit(passed, split, 'pos', 'no_shared', 4096))} | "
                           f"{diff(unit(passed, split, 'name', 'shared_joint', 4096), unit(passed, split, 'name', 'no_shared', 4096))} |")
            out.append("")
    out += ["## Per model, shared n=4096, 4-way (mean over folds; min-max over folds for seeds 6-9)", "",
            "| split | cand | " + " | ".join(f"seed {t}" for t in range(1, 10)) + " |", "|---|---|" + "---|" * 9]
    for split in SPLITS:
        for cand in CANDS:
            cells = []
            for t in range(1, 10):
                v = per[(False, split, cand, "shared_joint", 4096, "iia4")].get(t, [])
                cells.append("" if not v else f"{st.mean(v):.3f}" + (f" ({min(v):.2f}-{max(v):.2f})" if len(v) > 1 else ""))
            out.append(f"| {split} | {cand} | " + " | ".join(cells) + " |")
    (root / "extra" / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "extra" / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
