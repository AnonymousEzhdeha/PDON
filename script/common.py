"""Shared machinery for the held-out test stage.

The training scripts select `best_model.pth` on their evaluation split and never
reload it, so the number they report is the selection statistic itself. This
package supplies the missing step: load each saved checkpoint and score it on
freshly drawn test sets (same generators and settings as training, different
random draws).

Nothing here modifies any file of the original code base. The training scripts are
not importable — importing one creates a run directory, hijacks stdout, parses our
argv and starts training — so the data-generation helpers they define are recovered
by parsing the source and executing only its function/class definitions.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Fresh test seeds live far from the training seeds (0..9) so no training sample is
# ever reused, while the generator and its settings stay identical. This is the third
# split of the protocol: train -> select on eval/validation -> score here.
TEST_SEED_BASE = 10_000_000
TEST_SEED_STRIDE = 1_000


def test_seed(train_seed: int, k: int) -> int:
    """Seed for the k-th fresh test set of a run trained with `train_seed`."""
    return TEST_SEED_BASE + TEST_SEED_STRIDE * int(train_seed) + int(k)


# ---------------------------------------------------------------------------
# Recovering helpers from non-importable training scripts
# ---------------------------------------------------------------------------
def load_defs(script_path: str, names: Sequence[str], extra_globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute only the imports and def/class statements of `script_path`.

    Everything at module level in the training scripts (run-dir creation, stdout
    redirection, argparse, data generation, the training loop) is dropped, so this
    is side-effect free apart from the module's own imports.
    """
    with open(script_path, "r") as handle:
        tree = ast.parse(handle.read(), filename=script_path)
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    tree.body = [node for node in tree.body if isinstance(node, keep)]
    ns: Dict[str, Any] = {"__name__": "_pdon_defs", "__file__": script_path}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(tree, script_path, "exec"), ns)  # noqa: S102 - deliberate, see docstring
    missing = [n for n in names if n not in ns]
    if missing:
        raise KeyError(f"{os.path.basename(script_path)}: could not recover {missing}")
    return {n: ns[n] for n in names}


def add_suite_to_path(suite_dir: str) -> None:
    """Put a suite directory first on sys.path so its model modules resolve.

    Each suite ships its own copy of the model modules (the Beltrami and
    Brusselator `PODDON_TGV.py` differ), so the matching directory must win.
    """
    suite_dir = os.path.abspath(suite_dir)
    while suite_dir in sys.path:
        sys.path.remove(suite_dir)
    sys.path.insert(0, suite_dir)


# ---------------------------------------------------------------------------
# The metric, byte-for-byte as the training scripts compute it
# ---------------------------------------------------------------------------
def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Global flattened relative L2 - not a per-sample mean."""
    return float(torch.norm((pred - target).reshape(-1)) / torch.norm(target.reshape(-1)))


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor, batch_size: int, device: str) -> float:
    """Batched forward, fp32 accumulation on CPU, no AMP - matches training eval."""
    model.eval()
    preds, labels = [], []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size].to(device)
        yb = Y[i : i + batch_size]
        out = model(xb)
        preds.append(out.detach().cpu())
        labels.append(yb.detach().cpu())
    return rel_l2(torch.cat(preds, 0), torch.cat(labels, 0))


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
class Run:
    """A completed training run: its config, checkpoint and identifying fields."""

    def __init__(self, run_dir: str, config: Dict[str, Any], ckpt_path: str):
        self.run_dir = run_dir
        self.config = config
        self.ckpt_path = ckpt_path
        exp = config.get("experiment_config") or config.get("args") or {}
        self.args: Dict[str, Any] = exp
        self.model: str = str(exp.get("model", "unknown"))
        self.seed: int = int(exp.get("SEED", exp.get("seed", -1)))
        details = config.get("model_details") or {}
        self.total_parameters: Optional[int] = details.get("total_parameters")

    @property
    def label(self) -> str:
        return f"{os.path.basename(self.run_dir)}|{self.model}|seed{self.seed}"

    def __repr__(self) -> str:
        return f"<Run {self.label}>"


def discover_runs(
    suite_dir: str,
    run_subdirs: Sequence[str] = ("run",),
    models: Optional[Iterable[str]] = None,
    only_seeds: Optional[Iterable[int]] = None,
) -> List[Run]:
    """Find run directories holding both a config.json and a best_model.pth."""
    wanted_models = set(models) if models else None
    wanted_seeds = {int(s) for s in only_seeds} if only_seeds else None
    found: List[Run] = []
    for sub in run_subdirs:
        base = os.path.join(suite_dir, sub)
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base), key=lambda s: (len(s), s)):
            run_dir = os.path.join(base, entry)
            cfg_path = os.path.join(run_dir, "config.json")
            ckpt = os.path.join(run_dir, "weights", "best_model.pth")
            if not (os.path.isfile(cfg_path) and os.path.isfile(ckpt)):
                continue
            try:
                with open(cfg_path) as handle:
                    cfg = json.load(handle)
            except (json.JSONDecodeError, OSError):
                continue
            run = Run(run_dir, cfg, ckpt)
            if wanted_models and run.model not in wanted_models:
                continue
            if wanted_seeds is not None and run.seed not in wanted_seeds:
                continue
            found.append(run)
    return found


def load_state(run: Run, model: torch.nn.Module, device: str) -> Dict[str, Any]:
    """Load a checkpoint's weights, asserting the rebuild really matches."""
    ckpt = torch.load(run.ckpt_path, map_location=device, weights_only=False)
    n_rebuilt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if run.total_parameters is not None and int(run.total_parameters) != n_rebuilt:
        raise RuntimeError(
            f"{run.label}: rebuilt model has {n_rebuilt} parameters but the run recorded "
            f"{run.total_parameters}. The architecture reconstruction is wrong; refusing to "
            f"report a number from it."
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return ckpt


def reported_error(ckpt: Dict[str, Any]) -> Optional[float]:
    """The selection statistic the training run reported, for side-by-side context."""
    for key in ("test_error", "test_rel_l2", "best_test_error"):
        if key in ckpt:
            try:
                return float(ckpt[key])
            except (TypeError, ValueError):
                continue
    return None


# ---------------------------------------------------------------------------
# POD/PCA basis cache
#
# The basis is fitted on a run's training data and is NOT stored in the
# checkpoint (PCA_Layer_* keeps it as a plain tensor attribute, so it never
# enters state_dict). Rebuilding it means regenerating that seed's training data,
# which is expensive - hence the cache.
# ---------------------------------------------------------------------------
def _cache_path(suite: str, key: Dict[str, Any]) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = "_".join(f"{k}{key[k]}" for k in sorted(key))
    stamp = re.sub(r"[^A-Za-z0-9_.-]", "", stamp)
    return os.path.join(CACHE_DIR, f"pod_{suite}_{stamp}.npz")


def pod_cache_get(suite: str, key: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    path = _cache_path(suite, key)
    if not os.path.isfile(path):
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def pod_cache_put(suite: str, key: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    path = _cache_path(suite, key)
    tmp = f"{path}.tmp{os.getpid()}"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)  # atomic, so concurrent array tasks cannot see a partial file


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
ROW_FIELDS = [
    "suite",
    "model",
    "train_seed",
    "run_dir",
    "test_seed",
    "n_test",
    "rel_l2",
    "val_error_selection",
]


class Results:
    """Collects per-evaluation rows and writes a CSV plus a mean/std summary."""

    def __init__(self, suffix: str = ""):
        self.rows: List[Dict[str, Any]] = []
        self.suffix = f"_{suffix}" if suffix else ""

    def add(self, **row: Any) -> None:
        self.rows.append({k: row.get(k) for k in ROW_FIELDS})

    def summary(self) -> List[Dict[str, Any]]:
        groups: Dict[tuple, List[float]] = {}
        reported: Dict[tuple, List[float]] = {}
        for r in self.rows:
            if r["rel_l2"] is None:
                continue
            key = (r["suite"], r["model"])
            groups.setdefault(key, []).append(float(r["rel_l2"]))
            if r["val_error_selection"] is not None:
                reported.setdefault(key, []).append(float(r["val_error_selection"]))
        out = []
        for (suite, model), vals in sorted(groups.items()):
            rep = reported.get((suite, model), [])
            out.append(
                {
                    "suite": suite,
                    "model": model,
                    "n_evaluations": len(vals),
                    "heldout_mean": statistics.fmean(vals),
                    "heldout_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "heldout_min": min(vals),
                    "heldout_max": max(vals),
                    "val_selection_mean": statistics.fmean(rep) if rep else None,
                }
            )
        return out

    def write(self) -> Dict[str, str]:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        csv_path = os.path.join(RESULTS_DIR, f"test_results{self.suffix}.csv")
        json_path = os.path.join(RESULTS_DIR, f"summary{self.suffix}.json")
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(self.rows)
        with open(json_path, "w") as handle:
            json.dump(self.summary(), handle, indent=2)
        return {"csv": csv_path, "json": json_path}


def print_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("  (no evaluations)")
        return
    print(f"  {'suite':<14}{'model':<16}{'n':>4}  {'held-out mean +/- std':<26}{'val (selection)':>16}")
    for r in rows:
        rep = "-" if r["val_selection_mean"] is None else f"{r['val_selection_mean']:.4e}"
        print(
            f"  {r['suite']:<14}{r['model']:<16}{r['n_evaluations']:>4}  "
            f"{r['heldout_mean']:.4e} +/- {r['heldout_std']:.2e}   {rep:>12}"
        )


def parse_seed_list(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    out: List[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if part:
            out.append(int(part))
    return out or None


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
