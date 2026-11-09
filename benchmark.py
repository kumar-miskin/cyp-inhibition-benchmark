"""Post-challenge CYP regression benchmark. No challenge submissions are made."""

import argparse, copy, hashlib, json, platform, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import sklearn
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
ENDPOINTS = [
    x + "_pIC50_direct_inhibition" for x in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
]
SEED = 20261109


def download():
    dest = ROOT / "data"
    dest.mkdir(exist_ok=True)
    manifest_path = ROOT / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        info = json.load(
            urllib.request.urlopen(
                "https://huggingface.co/api/datasets/openadmet/cyp-challenge-train-test"
            )
        )
        manifest = {
            "dataset": "openadmet/cyp-challenge-train-test",
            "revision": info["sha"],
            "license": info.get("cardData", {}).get("license"),
            "files": {},
        }
    for part in ["TRAIN", "TEST-UNBLINDED"]:
        name = f"cyp-challenge-{part}_inhibition.csv"
        path = dest / name
        url = f"https://huggingface.co/datasets/{manifest['dataset']}/resolve/{manifest['revision']}/{name}"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        old = manifest["files"].get(name, {})
        if old and old["sha256"] != digest:
            raise ValueError("Dataset checksum changed: " + name)
        manifest["files"][name] = {
            "url": url,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dest, manifest


def mol_key(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError("Invalid SMILES: " + smiles)
    return Chem.MolToSmiles(m, isomericSmiles=True)


def scaffold_key(smiles):
    m = Chem.MolFromSmiles(smiles)
    # All ring-free compounds share the empty scaffold group, conservatively.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def split_indices(smiles, mode="scaffold", seed=SEED):
    groups = [scaffold_key(s) if mode == "scaffold" else mol_key(s) for s in smiles]
    train, val = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
            smiles, groups=groups
        )
    )
    assert not (set(np.array(groups)[train]) & set(np.array(groups)[val]))
    return train, val


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows = []
    for s in smiles:
        a = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(s)), a)
        rows.append(a)
    return np.stack(rows)


def st_rae(y, p, low, high):
    def err(q):
        return np.maximum(low - q, 0) + np.maximum(q - high, 0)

    denominator = err(np.full_like(y, np.mean(y))).sum()
    return float(err(p).sum() / denominator) if denominator > 0 else None


def score(df, pred):
    out = {}
    for i, c in enumerate(ENDPOINTS):
        good = df[c].notna().to_numpy()
        y = df.loc[good, c].to_numpy()
        p = pred[good, i]
        lo = df.loc[good, c + "_conf_low"].to_numpy()
        hi = df.loc[good, c + "_conf_high"].to_numpy()
        if not (
            np.isfinite(p).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
        ):
            raise ValueError("Non-finite score inputs")
        if np.any(lo > hi):
            raise ValueError("Reversed credible intervals")
        rho = (
            float(spearmanr(y, p).statistic)
            if np.ptp(p) > 1e-12 and np.ptp(y) > 1e-12
            else 0.0
        )
        out[c] = {
            "n": len(y),
            "MAE": float(mean_absolute_error(y, p)),
            "R2": float(r2_score(y, p)),
            "Spearman": rho,
            "ST-RAE": st_rae(y, p, lo, hi),
        }
    out["macro"] = {
        m: float(np.mean([out[c][m] for c in ENDPOINTS]))
        for m in ["MAE", "R2", "Spearman", "ST-RAE"]
    }
    return out


def atom_features(a):
    return [float(a.GetAtomicNum() == z) for z in [6, 7, 8, 9, 15, 16, 17, 35, 53]] + [
        a.GetAtomicNum() / 100,
        a.GetDegree() / 6,
        a.GetFormalCharge() / 4,
        float(a.GetIsAromatic()),
        a.GetTotalNumHs() / 4,
        float(a.IsInRing()),
    ]


def graph(smiles):
    m = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(a) for a in m.GetAtoms()], dtype=torch.float32)
    edges = []
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    e = (
        torch.tensor(edges, dtype=torch.long).T
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return x, e


def batch_graph(gs):
    xs = []
    es = []
    ids = []
    offset = 0
    for i, (x, e) in enumerate(gs):
        xs.append(x)
        es.append(e + offset)
        ids.append(torch.full((len(x),), i, dtype=torch.long))
        offset += len(x)
    return torch.cat(xs), torch.cat(es, dim=1), torch.cat(ids)


class MolecularGNN(nn.Module):
    """Small atom-message GNN. Bonds define adjacency; no bond-type embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(15, 64)
        self.selfs = nn.ModuleList([nn.Linear(64, 64) for _ in range(2)])
        self.neighbors = nn.ModuleList(
            [nn.Linear(64, 64, bias=False) for _ in range(2)]
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x, e, ids):
        h = torch.relu(self.embed(x))
        for s, n in zip(self.selfs, self.neighbors):
            agg = torch.zeros_like(h)
            degree = torch.zeros((len(h), 1), dtype=h.dtype)
            if e.shape[1]:
                agg.index_add_(0, e[1], h[e[0]])
                degree.index_add_(0, e[1], torch.ones((e.shape[1], 1)))
            h = torch.relu(s(h) + n(agg / degree.clamp(min=1)))
        pooled = torch.zeros((int(ids.max()) + 1, h.shape[1]))
        pooled.index_add_(0, ids, h)
        counts = torch.bincount(ids).float().sqrt().unsqueeze(1)
        return self.head(pooled / counts)


def masked_mse(p, y):
    mask = torch.isfinite(y)
    if not mask.any():
        raise ValueError("No observed labels")
    return ((p[mask] - y[mask]) ** 2).mean()


def gnn_fit(gs, y, epochs, validation=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = MolecularGNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    means = np.nanmean(y, axis=0)
    stds = np.nanstd(y, axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    targets = torch.tensor((y - means) / stds, dtype=torch.float32)
    best = None
    best_loss = float("inf")
    best_epoch = epochs
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(gs))
        loss_sum = 0
        for start in range(0, len(gs), 64):
            ix = order[start : start + 64]
            opt.zero_grad()
            p = model(*batch_graph([gs[i] for i in ix]))
            loss = masked_mse(p, targets[ix])
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(ix)
        row = {"epoch": epoch, "train_masked_mse": loss_sum / len(gs)}
        if validation:
            vg, vy = validation
            vp = gnn_predict(model, vg, means, stds)
            mask = np.isfinite(vy)
            vloss = float(np.mean(((vp - vy) / stds)[mask] ** 2))
            row["validation_standardized_mse"] = vloss
            if vloss < best_loss:
                best_loss = vloss
                best_epoch = epoch
                best = copy.deepcopy(model.state_dict())
        history.append(row)
    if best is not None:
        model.load_state_dict(best)
    return model, means, stds, best_epoch, history


def gnn_predict(model, gs, means, stds):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(gs), 128):
            out.append(model(*batch_graph(gs[i : i + 128])).numpy() * stds + means)
    return np.concatenate(out)


def run(epochs=40):
    torch.set_num_threads(2)
    np.random.seed(SEED)
    dest, manifest = download()
    train = pd.read_csv(dest / "cyp-challenge-TRAIN_inhibition.csv")
    hold = pd.read_csv(dest / "cyp-challenge-TEST-UNBLINDED_inhibition.csv")
    train["canonical"] = train.SMILES.map(mol_key)
    hold["canonical"] = hold.SMILES.map(mol_key)
    overlap = set(train.canonical) & set(hold.canonical)
    # Structure-only exclusion, never based on assay values.
    original_rows = len(train)
    train = train.loc[~train.canonical.isin(overlap)].reset_index(drop=True)
    audit = {
        "original_train_rows": original_rows,
        "holdout_rows": len(hold),
        "canonical_overlap_removed_from_training": len(overlap),
        "training_rows_after_exclusion": len(train),
        "train_duplicate_canonical_rows": int(train.canonical.duplicated().sum()),
        "holdout_duplicate_canonical_rows": int(hold.canonical.duplicated().sum()),
    }
    results = {
        "seed": SEED,
        "epochs_budget": epochs,
        "data_manifest": manifest,
        "audit": audit,
        "versions": {
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "validation": {},
        "original_holdout": {},
        "interpretation": "Post-challenge exploratory benchmark, not blind competition performance. No test-label tuning.",
    }
    x = fingerprints(train.SMILES)
    xh = fingerprints(hold.SMILES)
    y = train[ENDPOINTS].to_numpy()
    graphs = [graph(s) for s in train.SMILES]
    hg = [graph(s) for s in hold.SMILES]
    splits = {}
    for mode in ["random_grouped", "scaffold"]:
        a, b = split_indices(
            train.SMILES.to_list(), "scaffold" if mode == "scaffold" else "canonical"
        )
        splits[mode] = {"train": a.tolist(), "validation": b.tolist()}
        results["validation"][mode] = {
            "n_train": len(a),
            "n_validation": len(b),
            "models": {},
        }
        for model_name in ["mean", "morgan_rf"]:
            p = np.zeros((len(b), 4))
            for j, c in enumerate(ENDPOINTS):
                good = np.isfinite(y[a, j])
                ix = a[good]
                if model_name == "mean":
                    p[:, j] = np.mean(y[ix, j])
                else:
                    model = RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=2,
                        max_features=0.5,
                        n_jobs=2,
                        random_state=SEED,
                    )
                    model.fit(x[ix], y[ix, j])
                    p[:, j] = model.predict(x[b])
            results["validation"][mode]["models"][model_name] = score(train.iloc[b], p)
    # GNN and epoch selection use ONLY the scaffold training/validation split.
    a = np.array(splits["scaffold"]["train"])
    b = np.array(splits["scaffold"]["validation"])
    model, means, stds, chosen, history = gnn_fit(
        [graphs[i] for i in a], y[a], epochs, ([graphs[i] for i in b], y[b])
    )
    results["validation"]["scaffold"]["models"]["atom_gnn"] = score(
        train.iloc[b], gnn_predict(model, [graphs[i] for i in b], means, stds)
    )
    results["selected_epoch"] = chosen
    results["gnn_history"] = history
    # Fixed models are now trained on all original training rows, then holdout is scored once.
    for name in ["mean", "morgan_rf"]:
        p = np.zeros((len(hold), 4))
        for j, c in enumerate(ENDPOINTS):
            good = np.isfinite(y[:, j])
            if name == "mean":
                p[:, j] = np.mean(y[good, j])
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=0.5,
                    n_jobs=2,
                    random_state=SEED,
                )
                model.fit(x[good], y[good, j])
                p[:, j] = model.predict(xh)
        results["original_holdout"][name] = score(hold, p)
        pd.DataFrame(p, columns=ENDPOINTS).assign(
            Molecule_Name=hold.Molecule_Name
        ).to_csv(ROOT / "results" / f"{name}_holdout_predictions.csv", index=False)
    model, means, stds, _, _ = gnn_fit(graphs, y, chosen)
    p = gnn_predict(model, hg, means, stds)
    results["original_holdout"]["atom_gnn"] = score(hold, p)
    pd.DataFrame(p, columns=ENDPOINTS).assign(Molecule_Name=hold.Molecule_Name).to_csv(
        ROOT / "results" / "atom_gnn_holdout_predictions.csv", index=False
    )
    pd.DataFrame(
        {"Molecule_Name": train.Molecule_Name, "canonical": train.canonical}
    ).to_csv(ROOT / "results" / "training_index.csv", index=False)
    (ROOT / "results" / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    (ROOT / "results" / "metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "audit": audit,
                "selected_epoch": chosen,
                "holdout": {
                    k: v["macro"] for k, v in results["original_holdout"].items()
                },
                "validation": {
                    k: {n: m["macro"] for n, m in v["models"].items()}
                    for k, v in results["validation"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    run(args.epochs)
