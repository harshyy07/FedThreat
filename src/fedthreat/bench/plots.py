"""Plots for the three standard experiments. Input: list of summary rows (see runner.summarise)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Row = Dict[str, Any]


def _method(r: Row) -> str:
    return r["mode"] if r["mode"] in ("local", "central") else r["aggregator"]


def _val(r: Row, metric: str):
    key = f"nonowner_{metric}" if r["mode"] == "local" else metric
    k = f"{key}_mean"
    return (r.get(k, np.nan), r.get(f"{key}_std", 0.0))


def headline(rows: List[Row], out: Path) -> None:
    alphas = sorted({r["alpha"] for r in rows})
    methods = []
    for r in rows:
        if _method(r) not in methods:
            methods.append(_method(r))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (metric, title) in zip(axes, [("heldout_recall_1fpr", "Recall on the HELD-OUT attack family @1% FPR"),
                                          ("pr_auc", "Overall PR-AUC (all attacks)")]):
        w = 0.8 / len(methods)
        for mi, m in enumerate(methods):
            ys, es = [], []
            for al in alphas:
                rs = [r for r in rows if _method(r) == m and r["alpha"] == al]
                v, e = _val(rs[0], metric) if rs else (np.nan, 0)
                ys.append(v); es.append(e)
            ax.bar(np.arange(len(alphas)) + mi * w, ys, w, yerr=es, label=m, capsize=2)
        ax.set_xticks(np.arange(len(alphas)) + 0.4 - w / 2); ax.set_xticklabels([f"alpha={a}" for a in alphas])
        ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Local-only (orgs that never saw the family) vs federated vs centralized", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "headline.png", dpi=140); plt.close(fig)


def dp(rows: List[Row], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    arms = sorted({(r["aggregator"], r["clip"]) for r in rows})
    for ax, (metric, title) in zip(axes, [("pr_auc", "PR-AUC"), ("recall_1fpr", "Recall @1% FPR")]):
        for arm in arms:
            rs = sorted([r for r in rows if (r["aggregator"], r["clip"]) == arm], key=lambda r: r["sigma"])
            xs = [r["epsilon_mean"] for r in rs]
            fin = [i for i, x in enumerate(xs) if np.isfinite(x)]
            ax.errorbar([xs[i] for i in fin], [rs[i][f"{metric}_mean"] for i in fin],
                        yerr=[rs[i][f"{metric}_std"] for i in fin], marker="o", capsize=2, label=f"{arm[0]} clip={arm[1]}")
            base = [r for r in rs if not np.isfinite(r["epsilon_mean"])]
            if base:
                ax.axhline(base[0][f"{metric}_mean"], ls=":", lw=1, color=ax.lines[-1].get_color())
        ax.set_xscale("log"); ax.set_xlabel("epsilon (delta=1e-5, org-level, all rounds); dotted = no DP")
        ax.set_ylabel(title); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Privacy / utility trade-off", fontsize=11); fig.tight_layout()
    fig.savefig(out / "dp_tradeoff.png", dpi=140); plt.close(fig)


def attacks(rows: List[Row], out: Path) -> None:
    atks = sorted({r["attack"] for r in rows if r["attack"] != "none"})
    aggs = []
    for r in rows:
        if r["aggregator"] not in aggs:
            aggs.append(r["aggregator"])
    base = {r["aggregator"]: r for r in rows if r["attack"] == "none"}
    fig, axes = plt.subplots(1, len(atks), figsize=(3.6 * len(atks), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, atk in zip(axes, atks):
        for agg in aggs:
            rs = sorted([r for r in rows if r["attack"] == atk and r["aggregator"] == agg], key=lambda r: r["num_malicious"])
            xs = [0] + [r["num_malicious"] for r in rs]
            ys = ([base[agg]["pr_auc_mean"]] if agg in base else [np.nan]) + [r["pr_auc_mean"] for r in rs]
            ax.plot(xs, ys, marker="o", label=agg)
        ax.set_title(atk, fontsize=10); ax.set_xlabel("# malicious orgs"); ax.grid(alpha=.3)
    axes[0].set_ylabel("PR-AUC"); axes[-1].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "attack_pr_auc.png", dpi=140); plt.close(fig)
    # backdoor success rate
    bd = [r for r in rows if r["attack"] == "backdoor"]
    if bd:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for agg in aggs:
            rs = sorted([r for r in bd if r["aggregator"] == agg], key=lambda r: r["num_malicious"])
            b0 = base.get(agg, {}).get("asr_mean", np.nan)
            ax.plot([0] + [r["num_malicious"] for r in rs], [b0] + [r["asr_mean"] for r in rs], marker="o", label=agg)
        ax.set_xlabel("# malicious orgs"); ax.set_ylabel("backdoor attack-success rate"); ax.grid(alpha=.3)
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out / "backdoor_asr.png", dpi=140); plt.close(fig)


def make(kind: str, rows: List[Row], out: Path) -> None:
    {"headline": headline, "dp": dp, "attacks": attacks}[kind](rows, out)
