"""Phase 7: End-to-End Pipeline Integration Experiment.
Combines:
  1. Heterogeneous Data Partitions (non-IID Dirichlet)
  2. Missing-Feature Masking
  3. Client-Side Differential Privacy (Norm Clipping C=1.0, Gaussian Noise sigma=1.0)
  4. Privacy Budget Accountant ((epsilon, delta) tracking)
  5. Active Byzantine Attacker (Org 1 executes Model Replacement poisoning)
  6. Cohort-based Pairwise Secure Aggregation (X25519 ECDH + Shamir Secret Sharing)
  7. Byzantine Robust Server Defense (Trimmed Mean across Cohort sums)
"""

from fedthreat.sim.run_simulation import FederatedSimulation
from fedthreat.aggregation.robust import TrimmedMeanAggregator
from fedthreat.aggregation.cohort import CohortRobustAggregator
from fedthreat.attacks.poisoning import ModelReplacementAttacker


def run_integrated_pipeline():
    print("=" * 80)
    print("FedThreat: Phase 7 - Unified End-to-End Integration Pipeline")
    print("=" * 80)
    print("Active Layers:")
    print("  [Layer 1] Canonical Event Schema & Missing-Feature Mask Vector")
    print("  [Layer 2] Non-IID Dirichlet Label Skew across 9 Organizations")
    print("  [Layer 3] Client-Side DP-SGD (L2 Clip C=1.0, Gaussian Noise sigma=1.0, update-level / org-level DP)")
    print("  [Layer 4] Privacy Budget Accountant (Delta = 1e-5)")
    print("  [Layer 5] Active Adversary: Org 1 injects Model Replacement Poisoning")
    print("  [Layer 6] Secure Aggregation protocol code run in-process per cohort (SIMULATION, not a trust boundary)")
    print("  [Layer 7] Byzantine Defense: median across 3 random cohort aggregates")
    print("=" * 80)

    # 1. Initialize Simulation with 9 Organizations (3 cohorts of 3: the minimum for a robust median)
    sim = FederatedSimulation(
        num_orgs=9,
        alpha=0.5,
        held_out_attack="infiltration",
        held_out_org=0,
        seed=42,
    )
    print("[+] Preparing non-IID partitioned datasets...")
    sim.prepare_data(num_samples=12000)

    # 2. Configure Integrated Cohort Robust Aggregator
    base_robust_defense = TrimmedMeanAggregator(beta=0.2, min_trim=1)
    cohort_secagg_aggregator = CohortRobustAggregator(
        robust_aggregator=base_robust_defense,
        cohort_size=3,
    )

    # 3. Configure Active Attacker
    attacker = ModelReplacementAttacker(target_boost=4.0)
    malicious_map = {1: attacker}  # Organization 1 is adversarial

    # 4. Execute Unified Pipeline for 5 rounds
    print("\n[+] Launching 5 Federated Rounds with DP + SecAgg + Byzantine Defense...")
    results = sim.run_federated_rounds(
        num_rounds=5,
        local_epochs=2,
        clip_norm=1.0,
        noise_multiplier=1.0,
        target_delta=1e-5,
        max_epsilon=30.0,
        aggregator=cohort_secagg_aggregator,
        malicious_clients=malicious_map,
    )

    print("\n" + "-" * 80)
    print("Round-by-Round Integrated Results:")
    print(f"{'Round':<8} | {'Epsilon (eps)':<15} | {'PR-AUC':<10} | {'Macro-F1':<10} | {'Recall':<10}")
    print("-" * 80)
    for step in results["history"]:
        r_num = step["round"]
        eps = step["epsilon"]
        f1 = step["held_out_metrics"]["macro_f1"]
        pr_auc = step["held_out_metrics"]["pr_auc"]
        rec = step["held_out_metrics"]["macro_recall"]
        print(f"{r_num:<8} | {eps:<15.2f} | {pr_auc:<10.4f} | {f1:<10.4f} | {rec*100:<9.1f}%")

    print("-" * 80)
    print(f"Final Privacy Spend: eps = {results['final_epsilon']:.2f} (delta = 1e-5)")
    print(f"Global Model PR-AUC: {results['history'][-1]['held_out_metrics']['pr_auc']:.4f}")
    print("=" * 80)
    print("[SUCCESS] Phase 7 Unified Pipeline Execution Complete!")


if __name__ == "__main__":
    run_integrated_pipeline()
