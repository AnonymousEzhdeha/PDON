#!/usr/bin/env python
"""Run the held-out test stage across every experiment and aggregate the results.

Invoked automatically by run_all_experiments.sh (this is 
a native .sh file called to run our native cluster nodes. Users need to make one
putting config of their personal clusters to connect the infrastructure) once the seed sweep finishes, and
usable standalone. Experiments with no checkpoints are reported and skipped rather
than treated as errors.
"""
from __future__ import annotations

import argparse
import sys
import time

from common import Results, default_device, print_summary
from runner import run_suite
from suites import ORDER, REGISTRY


def main() -> int:
    p = argparse.ArgumentParser(description="Held-out test stage for all experiments")
    p.add_argument("--num_test_seeds", type=int, default=20)
    p.add_argument("--models", type=str, default="")
    p.add_argument("--only-seeds", dest="only_seeds", type=str, default="")
    p.add_argument("--suites", type=str, default="", help=f"subset of: {' '.join(ORDER)}")
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--out-suffix", dest="out_suffix", type=str, default="")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--limit-runs", dest="limit_runs", type=int, default=0)
    args = p.parse_args()

    names = [s for s in args.suites.replace(",", " ").split() if s] or ORDER
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown suite(s): {unknown}; known: {' '.join(ORDER)}", file=sys.stderr)
        return 2

    # One shared Results object so every experiment lands in a single CSV/summary.
    results = Results(args.out_suffix)
    forwarded = [
        "--num_test_seeds", str(args.num_test_seeds),
        "--device", args.device,
        "--out-suffix", args.out_suffix,
    ]
    if args.models:
        forwarded += ["--models", args.models]
    if args.only_seeds:
        forwarded += ["--only-seeds", args.only_seeds]
    if args.dry_run:
        forwarded += ["--dry-run"]
    if args.limit_runs:
        forwarded += ["--limit-runs", str(args.limit_runs)]

    t0 = time.time()
    for name in names:
        suite = REGISTRY[name]
        # each suite keeps its own n_test default, so do not forward --n_test here
        run_suite(suite, forwarded, results=results)

    paths = results.write()
    print(f"\n=== held-out test summary ({time.time() - t0:.0f}s) ===")
    print_summary(results.summary())
    print(f"\nrows: {paths['csv']}\nsummary: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
