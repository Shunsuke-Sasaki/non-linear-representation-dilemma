#!/usr/bin/env python3
"""Summary of ioi_namewhy.py (stdlib only). Usage: python3 ioi_namewhy_summary.py <EXP>

Tables, as fixed in the ioi_namewhy.py docstring:
  B   per seed 1-5 and restart: 4-way IIA of the individual namesplit maps on training / eval / test names.
  A1  per seed at L12: probe (unembed, embed; test names and held-out training-name prompts), logit lens, full END
      patching (src_io), and the namesplit C_name transfer (shared_joint n=4096, 4-way, mean over repeats and folds;
      results/ioi_ioname_stable/adapt and extra/adapt). Group means for {1,5,6,9} and {2,3,4,7,8}.
  A2  probe (unembed, test names) and patch src_io by layer, per seed.
  rho Spearman correlation over the 9 trained seeds between each L12 measure and the transfer.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

G1, G2 = [1, 5, 6, 9], [2, 3, 4, 7, 8]
SHOW_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 23]


def read_csv(p):
    with open(p) as f:
        return [{k: (float(v) if v not in ("trained", "random") else v) for k, v in r.items()} for r in csv.DictReader(f)]


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def transfer(root):
    reps = defaultdict(list)
    files = sorted((root / "adapt").glob("*.jsonl")) + sorted((root / "extra" / "adapt").glob("*.jsonl"))
    for f in files:
        for x in f.read_text().splitlines():
            r = json.loads(x)
            if (r["split"], r["trained_on"], r["condition"], r["adapt_examples"]) == ("namesplit", "name", "shared_joint", 4096):
                reps[(r.get("target", r["heldout"]), r["fold"])].append(r["name_iia4"])
    per = defaultdict(list)
    for (t, _), v in reps.items():
        per[t].append(st.mean(v))
    return {t: st.mean(v) for t, v in per.items()}


def main():
    exp = Path(sys.argv[1])
    out_dir = exp / "results" / "ioi_namewhy"
    stable = exp / "results" / "ioi_ioname_stable"
    lens = read_csv(out_dir / "lens_probe.csv")
    patch = read_csv(out_dir / "patch.csv")
    maps = [json.loads(x) for x in (out_dir / "maps_by_names.jsonl").read_text().splitlines()]
    tr = transfer(stable)
    L = {(r["state"], int(r["seed"]), int(r["layer"])): r for r in lens}
    P = {(r["state"], int(r["seed"]), int(r["layer"])): r for r in patch}
    out = ["# Why some seeds fit namesplit C_name (ioi_namewhy)", "", "Design and predictions: docstring of code/ioi_namewhy.py.", ""]

    out += ["## B. Individual namesplit maps on training / eval / test names (4-way; vocabulary in brackets)", "",
            "| seed | map | restart | rollbacks | train names | eval names | test names |", "|---|---|---|---|---|---|---|"]
    for r in sorted(maps, key=lambda r: (r["seed"], r["trained_on"] != "name", r["restart"])):
        c = r["trained_on"]
        cell = lambda s: f"{r[f'{s}_{c}_iia4']:.3f} [{r[f'{s}_{c}_iia_vocab']:.3f}]"
        out.append(f"| {r['seed']} | C_{c} | {r['restart']} | {r['rollbacks']} | {cell('train_names')} | "
                   f"{cell('eval_names')} | {cell('test_names')} |")

    cols = [("probe_unembed_test", "probe unembed, test names (chance .05)"),
            ("probe_unembed_seen", "probe unembed, training names (chance .017)"),
            ("probe_embed_test", "probe embed, test names"),
            ("lens_io_top_names", "lens: IO top among names"),
            ("patch_src_io", "full END patching: src_io")]

    def val(state, seed, layer, k):
        if k == "patch_src_io":
            return P[(state, seed, layer)]["pick_src_io"]
        return L[(state, seed, layer)][k]

    out += ["", "## A1. Layer 12 (the map site) per model", "",
            "| model | " + " | ".join(n for _, n in cols) + " | namesplit C_name transfer |", "|---|" + "---|" * (len(cols) + 1)]
    seeds = sorted({s for (st_, s, _) in L if st_ == "trained"})
    for s in seeds:
        out.append(f"| seed {s} | " + " | ".join(f"{val('trained', s, 12, k):.3f}" for k, _ in cols) +
                   f" | {tr.get(s, float('nan')):.3f} |")
    for s in sorted({s for (st_, s, _) in L if st_ == "random"}):
        out.append(f"| seed {s} step0 | " + " | ".join(f"{val('random', s, 12, k):.3f}" for k, _ in cols) + " | |")
    for name, g in (("{1,5,6,9}", G1), ("{2,3,4,7,8}", G2)):
        g = [s for s in g if s in seeds]
        out.append(f"| mean {name} | " + " | ".join(f"{st.mean(val('trained', s, 12, k) for s in g):.3f}" for k, _ in cols) +
                   f" | {st.mean(tr[s] for s in g if s in tr):.3f} |")

    out += ["", "Spearman rho over the trained seeds, L12 measure vs namesplit C_name transfer:", ""]
    xs = [s for s in seeds if s in tr]
    for k, n in cols:
        out.append(f"- {n}: rho = {spearman([val('trained', s, 12, k) for s in xs], [tr[s] for s in xs]):.2f} (k={len(xs)})")

    for k, n in (("probe_unembed_test", "probe unembed, test names"), ("patch_src_io", "full END patching: src_io")):
        out += ["", f"## A2. {n}, by layer", "", "| model | " + " | ".join(f"L{l}" for l in SHOW_LAYERS) + " |",
                "|---|" + "---|" * len(SHOW_LAYERS)]
        for state, s in [("trained", s) for s in seeds] + [("random", s) for s in sorted({s for (a, s, _) in L if a == "random"})]:
            out.append(f"| seed {s}{' step0' if state == 'random' else ''} | " +
                       " | ".join(f"{val(state, s, l, k):.2f}" for l in SHOW_LAYERS) + " |")
    text = "\n".join(out) + "\n"
    (out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
