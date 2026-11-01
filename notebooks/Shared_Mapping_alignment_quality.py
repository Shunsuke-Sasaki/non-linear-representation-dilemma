"""Quality of the per-model alignment A_m alone (no phi), on fresh normal inputs."""
import json, os, sys
import torch
REPO = "/home/sasaki/code_lib/causal_abstraction/non-linear-representation-dilemma"
sys.path.insert(0, REPO)
from das.Dataset_Generation import make_model_dataset
from das.Shared_Mapping import hidden_at, estimate_alignment, seed_everything_for, new_model, freeze

root = f"{REPO}/Results/SharedMapping/full"
cfg = json.load(open(f"{root}/config.json"))
dev = "cuda"
X_cal = torch.load(f"{root}/data/calibration.pt").to(dev)
seed_everything_for(20261014)  # fresh normal inputs, not used for estimating A_m
X_ev, _ = make_model_dataset(10000, 4, dev)

def load(seed, state):
    m = new_model(16).to(dev); m.load_state_dict(torch.load(f"{root}/models/seed{seed}.pt", map_location=dev)[state]); return freeze(m)

def r2(pred, target):  # variance of target explained, pooled over units (centred)
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()

def unit_corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    c = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)
    alive = (a.norm(dim=0) > 1e-6) & (b.norm(dim=0) > 1e-6)
    return c[alive]

rows = []
seeds = cfg["main_seeds"]
for state in ["trained", "random"]:
    for ref in [4287, 3837]:
        R = load(ref, state)
        H_ref_cal, H_ref = hidden_at(R, X_cal, 0), hidden_at(R, X_ev, 0)
        for s in seeds:
            if s == ref: continue
            M = load(s, state)
            H_cal, H = hidden_at(M, X_cal, 0), hidden_at(M, X_ev, 0)
            row = {"state": state, "ref": ref, "target": s,
                   "dead_ref": int((H_ref.std(0) < 1e-6).sum()), "dead_target": int((H.std(0) < 1e-6).sum())}
            row["identity_r2"] = r2(H, H_ref)
            for kind in ["permutation", "orthogonal"]:
                A, _ = estimate_alignment(R, M, X_cal, 0, kind)
                Z = A(H)
                row[f"{kind}_r2"] = r2(Z, H_ref)
                c = unit_corr(Z, H_ref)
                row[f"{kind}_corr_mean"] = c.mean().item(); row[f"{kind}_corr_min"] = c.min().item()
            # upper reference: unconstrained affine least squares fitted on calibration inputs
            Xc = torch.cat([H_cal, torch.ones(len(H_cal), 1, device=dev)], 1).double()
            W = torch.linalg.lstsq(Xc, H_ref_cal.double()).solution
            row["affine_lstsq_r2"] = r2((torch.cat([H, torch.ones(len(H), 1, device=dev)], 1).double() @ W).float(), H_ref)
            rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
cols = ["identity_r2", "permutation_r2", "orthogonal_r2", "affine_lstsq_r2", "permutation_corr_mean", "orthogonal_corr_mean", "permutation_corr_min"]
print(df[["state", "ref", "target"] + cols + ["dead_ref", "dead_target"]].round(3).to_string(index=False))
print()
print(df.groupby("state")[cols].agg(["mean", "min", "max"]).round(3).T.to_string())
df.to_csv(f"{root}/alignment_quality.csv", index=False)
