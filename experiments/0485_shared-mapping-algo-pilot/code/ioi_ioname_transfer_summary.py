#!/usr/bin/env python3
"""Tables of IOI phase 6 / 7 (transfer of C_name vs C_pos), stdlib only. Phase 7: results subdir ioi_ioname_stable.
Usage: python3 ioi_ioname_transfer_summary.py <EXP> [results subdir]"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

SIZES = [0, 256, 1024, 4096]


def ms(v):
    if not v:
        return ""
    return f"{st.mean(v):.3f}" if len(v) == 1 else f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"


def main():
    exp = Path(sys.argv[1])
    root = exp / "results" / (sys.argv[2] if len(sys.argv) > 2 else "ioi_ioname_transfer")
    rows = [json.loads(x) for f in sorted((root / "adapt").glob("*.jsonl")) for x in f.read_text().splitlines()]
    val = defaultdict(list)  # (split, cand, cond, n, held, metric) -> values over repeats
    for r in rows:
        for m in ("iia4", "iia_vocab"):
            val[(r["split"], r["trained_on"], r["condition"], r["adapt_examples"], r["heldout"], m)].append(
                r[f"{r['trained_on']}_{m}"])
    helds = sorted({r["heldout"] for r in rows})
    out = ["# IOI phase 6: transfer of C_pos and C_name (4-way IIA on the held-out model)", "",
           "Layer 12 END, revnet8, trained seeds. Held-out model: A_m adapted only (Procrustes init). "
           "Cells: mean ± SD over repeats (new minibatch seed and n-subset); n=0 and individual_self are single runs. "
           "Chance 0.25. In brackets: vocabulary argmax.", ""]
    for split in ["default", "namesplit"]:
        out += [f"## {split}", "",
                "| held-out | cand | individual_self | no_shared n=4096 | shared n=0 | shared n=256 | shared n=1024 | shared n=4096 |",
                "|---|---|---|---|---|---|---|---|"]
        for h in helds + ["mean 1,2,5"]:
            hs = [1, 2, 5] if h == "mean 1,2,5" else [h]
            for cand in ["pos", "name"]:
                def cell(cond, n, m="iia4"):
                    per = [st.mean(val[(split, cand, cond, n, x, m)]) for x in hs if val[(split, cand, cond, n, x, m)]]
                    if len(hs) > 1:
                        return f"{st.mean(per):.3f}" if len(per) == len(hs) else ""
                    return ms(val[(split, cand, cond, n, hs[0], m)])
                cells = [cell("individual_self", None) + f" [{cell('individual_self', None, 'iia_vocab')}]",
                         cell("no_shared", 4096)] + [cell("shared_joint", n) for n in SIZES[:-1]] + [
                         cell("shared_joint", 4096) + f" [{cell('shared_joint', 4096, 'iia_vocab')}]"]
                out.append(f"| {h} | {cand} | " + " | ".join(cells) + " |")
        out.append("")
    picks = defaultdict(list)
    for r in rows:
        if r["condition"] == "shared_joint" and r["adapt_examples"] == 4096:
            picks[(r["split"], r["trained_on"], r["heldout"])].append(r)
    out += ["## Which name the held-out model picks (shared_joint, n=4096, mean over repeats)", "",
            "| split | trained on | held-out | base IO | base S | source IO | source S |", "|---|---|---|---|---|---|---|"]
    for k in sorted(picks):
        v = picks[k]
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(
            f"{st.mean(r['pick_' + x] for r in v):.3f}" for x in ["base_io", "base_s", "src_io", "src_s"]) + " |")
    sel_path = root / "selection.json"
    if sel_path.exists():  # phase 7: which restart was used and how well it fits its source models
        sel = json.loads(sel_path.read_text())
        out += ["", "## Maps used (phase 7): restart with the lowest source dev loss", "",
                "2-way test IIA on each source model of the chosen restart; passed = all >= 0.95. "
                "Other restarts: best dev loss / min source IIA.", "",
                "| split | cand | kind | idx | chosen | source IIA | passed | rollbacks | all restarts |",
                "|---|---|---|---|---|---|---|---|---|"]
        for v in sorted(sel, key=lambda v: (v["split"], v["cand"], v["kind"], v["idx"])):
            ch = next(r for r in v["runs"] if r["restart"] == v["chosen"])
            others = ", ".join(f"r{r['restart']} {r['best_dev_loss']:.3f}/{min(r['test_iia']):.3f}" for r in v["runs"])
            out.append(f"| {v['split']} | {v['cand']} | {v['kind']} | {v['idx']} | r{v['chosen']} | "
                       + " ".join(f"{x:.3f}" for x in ch["test_iia"]) + f" | {v['passed']} | {ch['rollbacks']} | {others} |")
    (root / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
