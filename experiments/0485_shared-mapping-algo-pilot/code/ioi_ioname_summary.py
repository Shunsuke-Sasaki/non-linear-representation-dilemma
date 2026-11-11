#!/usr/bin/env python3
"""Tables of IOI phase 5 (ioi_ioname.py), stdlib only. Usage: python3 ioi_ioname_summary.py <EXP> [results subdir]"""
import json
import sys
from pathlib import Path

PHIS = ["identity", "rotation", "revnet1", "revnet2", "revnet4", "revnet8"]
FOUR = ["base_io", "base_s", "src_io", "src_s"]


def main():
    exp = Path(sys.argv[1])
    root = exp / "results" / (sys.argv[2] if len(sys.argv) > 2 else "ioi_ioname")
    rows = [json.loads(x) for f in sorted((root / "eval").glob("*.jsonl")) for x in f.read_text().splitlines()]
    maps = {(r["state"], r["split"], r["layer"], r["phi"], r["trained_on"]): r for r in rows if "trained_on" in r}
    ident = {(r["state"], r["split"], r["layer"], r["n_var_dims"]): r for r in rows if "n_var_dims" in r}
    out = ["# IOI phase 5: C_pos vs C_name (4-way IIA on the test pairs)", "",
           "Cells: IIA of a phi trained on that candidate, argmax over {base IO, base S, source IO, source S} "
           "(chance 0.25). identity = no training (the same 512 dims). pos_zdiff = C_pos IIA on the pairs where "
           "Z_pos differs (answer = base S).", ""]
    keys = sorted({k[:3] for k in maps} | {k[:3] for k in ident})
    for state in ["trained", "random"]:
        for split in ["default", "namesplit"]:
            layers = sorted({k[2] for k in keys if k[:2] == (state, split)})
            if not layers:
                continue
            out += [f"## {state} / {split}", "",
                    "| layer | cand | " + " | ".join(PHIS) + " |", "|---|---|" + "---|" * len(PHIS)]
            for L in layers:
                for cand, col in [("pos", "pos_iia4"), ("pos_zdiff", "pos_iia4_zdiff"), ("name", "name_iia4")]:
                    cells = []
                    for phi in PHIS:
                        r = (ident.get((state, split, L, 512)) if phi == "identity"
                             else maps.get((state, split, L, phi, cand.split("_")[0])))
                        cells.append(f"{r[col]:.3f}" if r else "")
                    out.append(f"| {L} | {cand} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["## Which name the intervened model picks (phi trained on C_name)", "",
            "| state | split | layer | phi | " + " | ".join(FOUR) + " | top outside four | best update / updates |",
            "|---|---|---|---|" + "---|" * (len(FOUR) + 2)]
    for k in sorted(maps):
        r = maps[k]
        if k[4] != "name":
            continue
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(f"{r['pick_' + x]:.3f}" for x in FOUR)
                   + f" | {r['top_outside_four']:.3f} | {r['best_update']} / {r['updates']} |")
    out += ["", "## No training: identity (512 dims), full END patching (1024 dims), clean accuracy", "",
            "| state | split | layer | dims | pos | name | " + " | ".join(FOUR) + " | clean 2-way |",
            "|---|---|---|---|---|---|" + "---|" * (len(FOUR) + 1)]
    for k in sorted(ident):
        r = ident[k]
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {r['pos_iia4']:.3f} | {r['name_iia4']:.3f} | "
                   + " | ".join(f"{r['pick_' + x]:.3f}" for x in FOUR) + f" | {r['clean_acc2']:.3f} |")
    gpu_s = sum(r.get("map_seconds", 0) + r.get("eval_seconds", 0) for r in maps.values())
    out += ["", f"Maps: {len(maps)}, training + evaluation {gpu_s / 3600:.2f} GPU h (excluding loading)."]
    (root / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
