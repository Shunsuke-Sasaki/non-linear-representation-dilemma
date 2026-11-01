"""Shared non-linear alignment maps across a population of MLPs.

Implements the research plan "モデル間共有と未見モデルへの汎化に基づく非線形対応写像の評価":
z_m = phi(A_m h_m), where A_m is a per-model alignment estimated from normal activations only
(permutation by default, orthogonal Procrustes as sensitivity analysis) and phi is either
model-specific (individual condition) or shared by several models.

Unlike Distributed_Alignment_Search_MLP this module does not use forward hooks: the MLP is split
at the target layer into a front part (hidden_at) and a back part (head_from), which lets one
batch of intervention IDs be evaluated on several models with one phi.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

from .Classification_Model import MLPForClassification


##############################################################################
##  MLP split at the target layer                                           ##
##############################################################################

def hidden_at(model, x, layer):
    """Activation after block `layer` of MLPForClassification (0 = first hidden layer)."""
    x = model.mlp.dropout(x)
    for block in model.mlp.h[:layer + 1]:
        x = block(x)
    return x


def head_from(model, h, layer):
    """Logits computed from the activation after block `layer`."""
    for block in model.mlp.h[layer + 1:]:
        h = block(h)
    return model.score(h)


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


##############################################################################
##  MLP training (plan §4: Adam, lr 1e-3, batch 1024, <=20 epochs,          ##
##  stop after 3 epochs without dev-loss improvement, keep best)            ##
##############################################################################

@torch.no_grad()
def mlp_loss_acc(model, X, y, batch_size=10000):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct = 0.0, 0
    for i in range(0, X.shape[0], batch_size):
        out = model(X[i:i + batch_size])
        loss += criterion(out, y[i:i + batch_size].long()).item()
        correct += (out.argmax(1) == y[i:i + batch_size].long()).sum().item()
    return loss / X.shape[0], correct / X.shape[0]


def train_mlp(model, X_train, y_train, X_dev, y_dev, batch_size=1024, max_epochs=20, patience=3,
              lr=0.001, generator=None, log=print):
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=generator)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, -1, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.long())
            loss.backward()
            optimizer.step()
            total += loss.item()
        dev_loss, dev_acc = mlp_loss_acc(model, X_dev, y_dev)
        history.append({"epoch": epoch + 1, "train_loss": total / len(loader),
                        "dev_loss": dev_loss, "dev_acc": dev_acc})
        log(f"  MLP epoch {epoch + 1}: train_loss={total / len(loader):.4f} "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}")
        if dev_loss < best_loss:
            best_loss, best_state, best_epoch, bad_epochs = dev_loss, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_dev_loss": best_loss, "history": history}


##############################################################################
##  Per-model alignment A_m (estimated from normal activations only)        ##
##############################################################################

class Alignment(nn.Module):
    """Invertible linear alignment h -> h @ Q. Permutations are applied by exact indexing."""

    def __init__(self, n, perm=None, matrix=None):
        super().__init__()
        self.kind = "identity" if perm is None and matrix is None else ("permutation" if perm is not None else "orthogonal")
        if perm is not None:
            perm = torch.as_tensor(perm, dtype=torch.long)
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", torch.argsort(perm))
        if matrix is not None:
            self.register_buffer("Q", torch.as_tensor(matrix, dtype=torch.float32))
        self.n = n

    def forward(self, h):
        if self.kind == "permutation":
            return h[:, self.perm]
        if self.kind == "orthogonal":
            return h @ self.Q
        return h

    def inverse(self, z):
        if self.kind == "permutation":
            return z[:, self.inv_perm]
        if self.kind == "orthogonal":
            return z @ self.Q.T
        return z

    def describe(self):
        if self.kind == "permutation":
            return {"kind": "permutation", "perm": self.perm.tolist()}
        if self.kind == "orthogonal":
            return {"kind": "orthogonal", "Q": self.Q.tolist()}
        return {"kind": "identity"}


def _standardize(H, eps=1e-8):
    mu = H.mean(0, keepdim=True)
    sd = H.std(0, keepdim=True)
    Z = (H - mu) / sd.clamp_min(eps)
    Z[:, (sd.squeeze(0) < eps)] = 0.0  # zero-variance (dead) units become identical all-zero columns
    return Z


def estimate_permutation(H_ref, H_m):
    """Plan §5.1: one-to-one assignment minimising the squared error of standardised activations.

    Returns perm with aligned[:, i] = H_m[:, perm[i]] matched to reference unit i, and the cost.
    Ties (e.g. several dead units) are resolved deterministically by scipy's solver on the
    float64 cost matrix. Standardisation is only used for estimation; A_m is the pure permutation.
    """
    Z_ref, Z_m = _standardize(H_ref.double()), _standardize(H_m.double())
    cost = ((Z_ref[:, :, None] - Z_m[:, None, :]) ** 2).mean(0).cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    assert (rows == np.arange(len(rows))).all()
    return torch.as_tensor(cols, dtype=torch.long), float(cost[rows, cols].sum())


def estimate_procrustes(H_ref, H_m):
    """Plan §5.2 (sensitivity analysis): uncentred orthogonal Procrustes, H_m Q ~ H_ref.

    No centring is applied, so A_m stays a pure linear map with inverse Q^T.
    """
    M = H_m.double().T @ H_ref.double()
    U, _, Vt = torch.linalg.svd(M)
    Q = (U @ Vt).float()
    cost = ((H_m @ Q - H_ref) ** 2).mean().item()
    return Q, cost


@torch.no_grad()
def estimate_alignment(model_ref, model_m, X_cal, layer, kind="permutation"):
    H_ref = hidden_at(model_ref, X_cal, layer)
    H_m = hidden_at(model_m, X_cal, layer)
    n = H_ref.shape[1]
    if kind == "permutation":
        perm, cost = estimate_permutation(H_ref, H_m)
        return Alignment(n, perm=perm).to(X_cal.device), cost
    if kind == "orthogonal":
        Q, cost = estimate_procrustes(H_ref, H_m)
        return Alignment(n, matrix=Q).to(X_cal.device), cost
    raise ValueError(kind)


##############################################################################
##  Intervention data                                                       ##
##############################################################################

class InterventionSet:
    """Tensors of an intervention dataset of the hierarchical equality task.

    base (N,16), sources (2,N,16), label (N,) counterfactual label, iv (N,2) bool,
    base_label (N,) high-level output for the un-intervened base.
    """

    def __init__(self, base, sources, label, iv, base_label):
        self.base, self.sources, self.label, self.iv, self.base_label = base, sources, label, iv, base_label

    @classmethod
    def from_raw(cls, raw, embedding_size=4):
        base = torch.stack([d["base"] for d in raw])
        sources = torch.stack([d["sources"] for d in raw], dim=1)
        label = torch.stack([d["label"] for d in raw]).long()
        iv = torch.tensor([d["intervention"] for d in raw], dtype=torch.bool)
        return cls(base, sources, label, iv, both_equality_label(base, embedding_size))

    def __len__(self):
        return self.base.shape[0]

    def subset(self, idx):
        return InterventionSet(self.base[idx], self.sources[:, idx], self.label[idx], self.iv[idx], self.base_label[idx])

    def to(self, device):
        return InterventionSet(*(t.to(device) for t in self.state()))

    def state(self):
        return (self.base, self.sources, self.label, self.iv, self.base_label)

    def save(self, path):
        torch.save(dict(zip(["base", "sources", "label", "iv", "base_label"], self.state())), path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location=device)
        return cls(d["base"], d["sources"], d["label"], d["iv"], d["base_label"])


def both_equality_label(x, embedding_size=4):
    e = embedding_size
    e_l = (x[:, 0:e] == x[:, e:2 * e]).all(1)
    e_r = (x[:, 2 * e:3 * e] == x[:, 3 * e:4 * e]).all(1)
    return (e_l == e_r).long()


##############################################################################
##  Interventions through A and phi                                         ##
##############################################################################

class IdentityMap(nn.Module):
    """phi = id: the 'alignment only' diagnostic condition."""

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


def variable_masks(n, variable_dims, device):
    masks = torch.zeros(len(variable_dims), n, dtype=torch.bool, device=device)
    for v, dims in enumerate(variable_dims):
        masks[v, dims] = True
    return masks


def intervene_logits(phi, target, data, variable_dims, layer, source=None):
    """Interchange intervention in the shared space.

    target / source are (model, alignment) pairs. With source=None this is the within-model
    intervention of plan §3; otherwise the source activations come from another model (§9):
    h~ = A_t^-1 phi^-1(Swap(phi(A_t h_t,b), phi(A_s h_s,src))).
    """
    model_t, align_t = target
    model_s, align_s = target if source is None else source
    B = data.base.shape[0]
    h_b = hidden_at(model_t, data.base, layer)
    h_src = hidden_at(model_s, data.sources.reshape(-1, data.sources.shape[-1]), layer)
    if source is None:
        z = phi(align_t(torch.cat([h_b, h_src])))
        z_b, z_src = z[:B], z[B:]
    else:
        z_b, z_src = phi(align_t(h_b)), phi(align_s(h_src))
    z_src = z_src.reshape(len(variable_dims), B, -1)
    masks = variable_masks(z_b.shape[1], variable_dims, z_b.device)
    z_new = z_b
    for v in range(len(variable_dims)):
        sel = data.iv[:, v:v + 1] & masks[v][None, :]
        z_new = torch.where(sel, z_src[v], z_new)
    h_new = align_t.inverse(phi.inverse(z_new))
    return head_from(model_t, h_new, layer)


@torch.no_grad()
def evaluate(phi, target, data, variable_dims, layer, source=None, batch_size=10000):
    """IIA overall and broken down by intervention type and by whether the label changes."""
    was_training = phi.training
    phi.eval()
    preds, losses = [], []
    for i in range(0, len(data), batch_size):
        sub = data.subset(slice(i, i + batch_size))
        logits = intervene_logits(phi, target, sub, variable_dims, layer, source)
        preds.append(logits.argmax(1))
        losses.append(nn.functional.cross_entropy(logits, sub.label, reduction="sum"))
    phi.train(was_training)
    pred = torch.cat(preds)
    correct = pred == data.label
    iv = data.iv
    groups = {
        "left": iv[:, 0] & ~iv[:, 1],
        "right": ~iv[:, 0] & iv[:, 1],
        "both": iv[:, 0] & iv[:, 1],
        "label_changed": data.label != data.base_label,
        "label_unchanged": data.label == data.base_label,
    }
    out = {"iia": correct.float().mean().item(), "loss": (torch.stack(losses).sum() / len(data)).item(), "n": len(data)}
    for name, mask in groups.items():
        out[f"iia_{name}"] = correct[mask].float().mean().item() if mask.any() else None
        out[f"n_{name}"] = int(mask.sum().item())
    return out, correct.cpu()


##############################################################################
##  Training phi on one (individual) or several (shared) models            ##
##############################################################################

def train_mapping(phi, members, train, dev, variable_dims, layer, ids_per_member=6400, max_epochs=50,
                  patience=5, improve_threshold=0.001, lr=0.001, scheduler_patience=10,
                  generator=None, log=print):
    """Train phi only; every model and every alignment stays frozen (plan §7.2).

    Each update draws `ids_per_member` intervention IDs and evaluates the *same* IDs on every
    member; the loss is the unweighted mean of the members' cross-entropies. One epoch is one pass
    over all intervention IDs, so each member sees the full training set once per epoch.
    Early stopping monitors the mean (over members) per-example dev loss and follows the
    repository's DAS rule: keep the best phi, reset patience only on an improvement > threshold.
    """
    for model, _ in members:
        freeze(model)
    optimizer = optim.Adam(phi.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=scheduler_patience)
    N = len(train)
    best_loss, best_loss_corr, best_state, bad = float("inf"), float("inf"), None, 0
    history, updates = [], 0
    for epoch in range(max_epochs):
        phi.train()
        order = torch.randperm(N, generator=generator).to(train.base.device)
        total, n_batches = 0.0, 0
        for start in range(0, N, ids_per_member):
            sub = train.subset(order[start:start + ids_per_member])
            optimizer.zero_grad()
            loss = torch.stack([
                nn.functional.cross_entropy(intervene_logits(phi, m, sub, variable_dims, layer), sub.label)
                for m in members]).mean()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1
            updates += 1
        scheduler.step(total / n_batches)
        dev_stats = [evaluate(phi, m, dev, variable_dims, layer)[0] for m in members]
        dev_loss = float(np.mean([s["loss"] for s in dev_stats]))
        if dev_loss < best_loss_corr:
            best_loss_corr, best_state = dev_loss, copy.deepcopy(phi.state_dict())
            if dev_loss + improve_threshold < best_loss:
                best_loss, bad = dev_loss, 0
            else:
                bad += 1
        else:
            bad += 1
        history.append({"epoch": epoch + 1, "train_loss": total / n_batches, "dev_loss": dev_loss,
                        "dev_iia_per_member": [s["iia"] for s in dev_stats],
                        "lr": optimizer.param_groups[0]["lr"], "updates": updates})
        log(f"  phi epoch {epoch + 1}: train_loss={total / n_batches:.4f} dev_loss={dev_loss:.4f} "
            f"dev_iia={[round(s['iia'], 4) for s in dev_stats]} bad={bad}")
        if bad >= patience:
            break
    phi.load_state_dict(best_state)
    phi.eval()
    return {"best_dev_loss": best_loss_corr, "epochs": len(history), "updates": updates,
            "examples_processed": updates * ids_per_member * len(members), "history": history}


##############################################################################
##  Controls (plan §10)                                                     ##
##############################################################################

def permuted_copy(model, layer, perm):
    """Functionally identical MLP whose units at `layer` are reordered: h_copy = h[:, perm]."""
    copy_model = copy.deepcopy(model)
    perm = torch.as_tensor(perm, dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        ff = copy_model.mlp.h[layer].ff1
        ff.weight.copy_(ff.weight[perm])
        ff.bias.copy_(ff.bias[perm])
        nxt = copy_model.mlp.h[layer + 1].ff1 if layer + 1 < len(copy_model.mlp.h) else copy_model.score
        nxt.weight.copy_(nxt.weight[:, perm])
    return copy_model


def seed_everything_for(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def new_model(input_size=16):
    return MLPForClassification(input_size=input_size)
