"""Generic evaluation loop shared by the seven per-experiment test scripts."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from typing import List, Optional

import torch

import common
from common import Results, default_device, discover_runs, load_state, parse_seed_list, reported_error, test_seed
from suites import REGISTRY, Suite


def build_parser(suite: Suite) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Held-out test evaluation for the {suite.name} experiment")
    p.add_argument("--num_test_seeds", type=int, default=20,
                   help="number of freshly drawn test sets per checkpoint")
    p.add_argument("--n_test", type=int, default=suite.default_n_test,
                   help="samples per fresh test set (0 = whole available split)")
    p.add_argument("--models", type=str, default="",
                   help="space/comma separated model filter, e.g. 'OSS Mamba'")
    p.add_argument("--only-seeds", dest="only_seeds", type=str, default="",
                   help="restrict to these training seeds (used to split work across array tasks)")
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--out-suffix", dest="out_suffix", type=str, default="")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="list discovered checkpoints and the planned matrix, evaluate nothing")
    p.add_argument("--limit-runs", dest="limit_runs", type=int, default=0,
                   help="evaluate at most this many checkpoints (0 = no limit)")
    return p


def run_suite(suite: Suite, argv: Optional[List[str]] = None, results: Optional[Results] = None) -> Results:
    args = build_parser(suite).parse_args(argv)
    own_results = results is None
    results = results or Results(args.out_suffix)

    ok, why = suite.available()
    if not ok:
        print(f"[{suite.name}] unavailable: {why}")
        if own_results:
            results.write()
        return results

    models = [m for m in args.models.replace(",", " ").split() if m] or None
    runs = discover_runs(suite.suite_dir, suite.run_subdirs, models, parse_seed_list(args.only_seeds))
    if args.limit_runs:
        runs = runs[: args.limit_runs]

    if not runs:
        print(f"[{suite.name}] no checkpoints found under "
              f"{', '.join(os.path.join(suite.directory, s) for s in suite.run_subdirs)} - nothing to test")
        if own_results:
            results.write()
        return results

    print(f"[{suite.name}] {len(runs)} checkpoint(s); {args.num_test_seeds} fresh test set(s) each, "
          f"n_test={args.n_test or 'full'}, device={args.device}")
    if args.dry_run:
        for run in runs:
            seeds = [test_seed(run.seed, k) for k in range(args.num_test_seeds)]
            print(f"  {run.label:<34} -> test seeds {seeds[0]}..{seeds[-1]}")
        if own_results:
            results.write()
        return results

    for run in runs:
        t0 = time.time()
        try:
            model = suite.build_model(run, args.device)
            ckpt = load_state(run, model, args.device)
        except Exception as exc:  # a wrong rebuild must be loud, never silently scored
            print(f"  {run.label:<34} REBUILD FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        selection = reported_error(ckpt)
        batch = suite.batch_size(run)
        values = []
        for k in range(args.num_test_seeds):
            seed = test_seed(run.seed, k)
            try:
                data = suite.make_test_data(run, seed, args.n_test)
                value = suite.evaluate(model, data, batch, args.device)
            except Exception as exc:
                print(f"  {run.label} test_seed={seed} FAILED: {type(exc).__name__}: {exc}")
                continue
            values.append(value)
            results.add(
                suite=suite.name,
                model=run.model,
                train_seed=run.seed,
                run_dir=os.path.relpath(run.run_dir, common.REPO_ROOT),
                test_seed=seed,
                n_test=int(data["X"].shape[0]),
                rel_l2=value,
                reported_selection_error=selection,
            )
            del data
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if values:
            mean = sum(values) / len(values)
            spread = max(values) - min(values)
            sel = "-" if selection is None else f"{selection:.4e}"
            print(f"  {run.label:<34} held-out {mean:.4e} (spread {spread:.2e}, n={len(values)})  "
                  f"selection {sel}  [{time.time() - t0:.0f}s]")

    if own_results:
        paths = results.write()
        print(f"[{suite.name}] wrote {paths['csv']}")
    return results


def main_for(name: str) -> None:
    run_suite(REGISTRY[name])
