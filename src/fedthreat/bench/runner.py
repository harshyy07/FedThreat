"""Config-driven benchmark runner.

  python -m fedthreat.bench.runner configs/bench/headline.yaml --out runs/headline --jobs 8

YAML:  name, plot (headline|dp|attacks), seeds, base (RunConfig defaults), runs: [{grid: {field: [values]}}]
Each grid is expanded as a cartesian product x seeds. Outputs: results.csv (one row per run),
summary.csv (mean/std over seeds), config.yaml snapshot, git-independent plots.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from fedthreat.bench.fl import RunConfig, run

CFG_FIELDS = [f.name for f in fields(RunConfig)]
METRICS = ["pr_auc", "recall_1fpr", "recall_0.1fpr", "heldout_recall_1fpr", "asr", "miss_clean", "epsilon"]


def expand(spec: Dict[str, Any]) -> List[RunConfig]:
    out: List[RunConfig] = []
    for entry in spec["runs"]:
        grid = entry["grid"]
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            for seed in spec.get("seeds", [0]):
                out.append(RunConfig(**{**spec.get("base", {}), **dict(zip(keys, combo)), "seed": seed,
                                        "experiment": spec["name"]}))
    return out


def _safe_run(cfg: RunConfig) -> Dict[str, Any]:
    try:
        return run(cfg)
    except Exception as exc:  # keep sweeps going; record the failure
        return {**cfg.__dict__, "error": repr(exc)}


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_keys = [k for k in CFG_FIELDS if k not in ("seed",)]
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        if "error" in r:
            continue
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    out = []
    mets: List[str] = []
    for r in rows:
        for m in r:
            if (m in METRICS or m.startswith("nonowner_")) and m not in mets:
                mets.append(m)
    for key, rs in groups.items():
        row = dict(zip(group_keys, key)); row["n_seeds"] = len(rs)
        for m in mets:
            vals = np.array([r.get(m, float("nan")) for r in rs], float)
            if np.isinf(vals).any() and not np.isfinite(vals).any():   # e.g. epsilon with DP off
                row[f"{m}_mean"], row[f"{m}_std"] = float("inf"), 0.0
            else:
                fin = vals[np.isfinite(vals)]
                row[f"{m}_mean"] = float(fin.mean()) if len(fin) else float("nan")
                row[f"{m}_std"] = float(fin.std()) if len(fin) else float("nan")
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--out", default=None); ap.add_argument("--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(open(a.config))
    out = Path(a.out or f"runs/{spec['name']}"); out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.yaml")
    cfgs = expand(spec)
    print(f"[{spec['name']}] {len(cfgs)} runs on {a.jobs} workers -> {out}", flush=True)
    t0 = time.time()
    with mp.Pool(a.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap(_safe_run, cfgs, chunksize=4), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(cfgs):
                print(f"  {i}/{len(cfgs)} ({time.time() - t0:.0f}s)", flush=True)
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"[!] {len(errs)} runs failed, first: {errs[0]['error']}", file=sys.stderr)
    write_csv(out / "results.csv", rows)
    summ = summarise(rows)
    write_csv(out / "summary.csv", summ)
    if not a.no_plots and spec.get("plot"):
        from fedthreat.bench import plots
        plots.make(spec["plot"], summ, out)
    print(f"done in {time.time() - t0:.0f}s: {out}/summary.csv")


if __name__ == "__main__":
    main()
