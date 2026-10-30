"""Phase 5 Evaluation: Defense Benchmark under Byzantine Poisoning Attacks.
Compares FedAvg against Coordinate-wise Median, Trimmed Mean, and Krum under attack.
"""

from fedthreat.sim.run_simulation import FederatedSimulation
from fedthreat.aggregation.fedavg import FedAvgAggregator
from fedthreat.aggregation.robust import (
    CoordinateWiseMedianAggregator,
    TrimmedMeanAggregator,
    KrumAggregator,
)
from fedthreat.attacks.poisoning import ModelReplacementAttacker, SignFlippingAttacker


def run_robustness_benchmark():
    print("=" * 75)
    print("FedThreat: Phase 5 - Byzantine Robust Aggregation & Attack Suite")
    print("=" * 75)
    print("Scenario:")
    print("  - 5 Organizations (1 Malicious Attacker: Org 1 performs Model Replacement)")
    print("  - Target: Model Replacement attack tries to destroy threat detection utility")
    print("=" * 75)

    aggregators = [
        ("FedAvg (Vulnerable Baseline)", FedAvgAggregator()),
        ("Coordinate-wise Median", CoordinateWiseMedianAggregator()),
        ("Trimmed Mean (beta=0.2)", TrimmedMeanAggregator(beta=0.2)),
        ("Multi-Krum (f=1, m=2)", KrumAggregator(num_malicious=1, m=2)),
    ]

    attacker = ModelReplacementAttacker(target_boost=5.0)
    malicious_map = {1: attacker}  # Org 1 is the Byzantine node

    results = []

    for name, agg in aggregators:
        print(f"\n[+] Testing {name} under Model Replacement attack (5 rounds)...")
        sim = FederatedSimulation(
            num_orgs=5,
            alpha=0.5,
            held_out_attack="infiltration",
            held_out_org=0,
            seed=42,
        )
        sim.prepare_data(num_samples=10000)

        out = sim.run_federated_rounds(
            num_rounds=5,
            local_epochs=2,
            aggregator=agg,
            malicious_clients=malicious_map,
        )

        final_step = out["history"][-1]
        rec = final_step["held_out_metrics"]["macro_recall"]
        f1 = final_step["held_out_metrics"]["macro_f1"]
        pr_auc = final_step["held_out_metrics"]["pr_auc"]

        results.append({
            "aggregator": name,
            "recall": rec,
            "macro_f1": f1,
            "pr_auc": pr_auc,
        })

    print("\n" + "=" * 75)
    print(f"{'Aggregator Strategy':<32} | {'Recall':<10} | {'Macro-F1':<10} | {'PR-AUC':<10}")
    print("-" * 75)
    for r in results:
        status = "COMPROMISED" if r["macro_f1"] < 0.1 else "RESILIENT"
        print(
            f"{r['aggregator']:<32} | {r['recall']*100:<9.1f}% | {r['macro_f1']:<10.4f} | {r['pr_auc']:<10.4f} [{status}]"
        )
    print("=" * 75)
    print("[SUCCESS] Phase 5 Robustness Benchmark Complete!")


if __name__ == "__main__":
    run_robustness_benchmark()
