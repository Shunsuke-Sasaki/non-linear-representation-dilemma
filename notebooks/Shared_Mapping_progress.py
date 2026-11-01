#!/usr/bin/env python
"""Progress of notebooks/Shared_Mapping.py, read from saved checkpoints and the log.

  .venv/bin/python notebooks/Shared_Mapping_progress.py            # once
  watch -n 30 .venv/bin/python notebooks/Shared_Mapping_progress.py  # live
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess

ALIGNS = ["permutation", "orthogonal"]
STATES = ["trained", "random"]


def count(path, pattern=r".*\.pt$"):
    if not os.path.isdir(path):
        return 0
    return sum(bool(re.match(pattern, f)) for f in os.listdir(path))


def bar(done, total, width=20):
    filled = int(width * done / total) if total else width
    return "#" * filled + "." * (width - filled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    root = os.path.abspath(os.path.join(repo, "Results", "SharedMapping", "quick" if args.quick else "full"))
    cfg = json.load(open(os.path.join(root, "config.json")))
    n_seeds, n_inits = len(cfg["main_seeds"]), cfg["n_inits"]
    log = open(os.path.join(root, "log.txt")).read().splitlines()

    # Mean seconds per saved map, by category ("individual", "<align>/shared_linear", ...).
    durations = {}
    for line in log:
        m = re.search(r"saved (\S+)\.pt \((\d+)s", line)
        if m:
            parts = m.group(1).split("/")
            if parts[0] == "dev":
                continue
            key = "individual" if parts[0] == "individual" else f"{parts[0]}/{parts[2]}"
            durations.setdefault(key, []).append(int(m.group(2)))

    def mean_secs(key):
        if key in durations:
            return sum(durations[key]) / len(durations[key])
        other = [v for k, v in durations.items() if k.endswith(key.split("/")[-1])]
        return sum(sum(o) for o in other) / max(1, sum(len(o) for o in other)) if other else None

    main_models = sum(os.path.exists(os.path.join(root, "models", f"seed{s}.pt")) for s in cfg["main_seeds"])
    stages = [("MLP models (main seeds)", main_models, n_seeds, None)]
    per_map = n_seeds * len(STATES) * n_inits
    stages.append(("individual phi", sum(count(os.path.join(root, "individual", s)) for s in STATES), per_map,
                   "individual"))
    for align in ALIGNS:
        for cond in ["shared_linear", "shared_nonlinear"]:
            done = sum(count(os.path.join(root, align, "maps", cond, s)) for s in STATES)
            stages.append((f"{align}: {cond}", done, per_map, f"{align}/{cond}"))
        stages.append((f"{align}: evaluate + summary", int(os.path.exists(os.path.join(root, align, "summary.md"))), 1,
                       None))

    procs = subprocess.run(["pgrep", "-af", "notebooks/Shared_Mapping.py"], capture_output=True, text=True).stdout
    running = [l.split("Shared_Mapping.py", 1)[1].strip() for l in procs.splitlines() if "python" in l.split()[1]]

    print(f"Shared mapping progress  ({dt.datetime.now():%Y-%m-%d %H:%M:%S}, {root})")
    print(f"process : {'RUNNING  ' + ' | '.join(running) if running else 'not running'}")
    print()
    remaining_secs = 0.0
    for name, done, total, key in stages:
        status = "done" if done >= total else ""
        avg = mean_secs(key) if key else None
        avg_txt = f"~{avg / 60:.1f} min/map" if avg else ""
        print(f"  {name:<34} [{bar(done, total)}] {done:>2}/{total:<2} {status:<4} {avg_txt}")
        if key and done < total and avg:
            remaining_secs += (total - done) * avg
    eval_left = sum(1 for name, done, total, _ in stages if "evaluate" in name and done < total)
    remaining_secs += eval_left * 120  # one evaluation pass took about 2 minutes

    current = next((l for l in reversed(log) if re.search(r"\] (shared_\w+: fold|Individual RevNet|Training MLP|Evaluating)", l)), None)
    last = log[-1] if log else ""
    print()
    print(f"current : {current.split('] ', 1)[1] if current else '-'}")
    m = re.search(r"phi epoch (\d+):.*dev_iia=(\[[^\]]*\]) bad=(\d+)", last)
    if m:
        print(f"          epoch {m.group(1)}/{cfg['map_max_epochs']}, dev IIA {m.group(2)}, "
              f"epochs without improvement {m.group(3)}/{cfg['map_patience']}")
    print(f"last log: {last[:150]}")
    if running and remaining_secs:
        eta = dt.datetime.now() + dt.timedelta(seconds=remaining_secs)
        print(f"ETA     : ~{remaining_secs / 3600:.1f} h left, around {eta:%H:%M} "
              f"(from mean time per finished map; random-state runs vary)")
    elif not running and any(done < total for _, done, total, _ in stages):
        print("WARNING : unfinished stages but no running process")


if __name__ == "__main__":
    main()
