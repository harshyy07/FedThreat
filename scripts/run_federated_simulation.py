"""Headline Experiment: Comparing Local-Only Training vs Federated Learning (FedAvg / FedProx)
on detecting rare/held-out attacks across organizations.
"""

from fedthreat.sim.run_simulation import FederatedSimulation


def run_experiment():
    print("=" * 70)
    print("FedThreat: Phase 2 - Federated Simulation & Held-Out Attack Experiment")
    print("=" * 70)
    print("Scenario:")
    print("  - 5 Organizations with non-IID Dirichlet distribution (alpha = 0.5)")
    print("  - Organization 0 is the ONLY organization that experiences 'infiltration'")
    print("  - Organizations 1, 2, 3, 4 have NEVER seen 'infiltration' during training")
    print("=" * 70)

    sim = FederatedSimulation(
        num_orgs=5,
        alpha=0.5,
        held_out_attack="infiltration",
        held_out_org=0,
        seed=42,
    )
    print("[+] Partitioning data across 5 virtual organizations...")
    sim.prepare_data(num_samples=15000)

    # 1. Local-Only Baseline
    print("\n[+] Training Local-Only models (no collaboration across orgs)...")
    local_results = sim.run_local_only_baseline(epochs_per_org=8)

    print("-" * 70)
    print("Local-Only Detection of 'infiltration' (held-out from Orgs 1-4):")
    for org_id, metrics in local_results.items():
        role = "Saw attack during training" if org_id == 0 else "NEVER saw attack"
        rec = metrics.get("macro_recall", 0.0)
        f1 = metrics.get("macro_f1", 0.0)
        print(f"  * Org {org_id} ({role:25s}) -> Recall: {rec*100:.1f}% | Macro-F1: {f1:.4f}")

    # 2. Federated Learning (FedAvg)
    print("\n[+] Running Federated Learning (FedAvg: 8 rounds, 2 local epochs/round)...")
    fed_results = sim.run_federated_rounds(num_rounds=8, local_epochs=2, mu_fedprox=0.0)

    final_metrics = fed_results["history"][-1]["held_out_metrics"]
    print("-" * 70)
    print("Federated Global Model Detection of 'infiltration' across ALL orgs:")
    print(f"  * Global Model Recall on Held-Out Attack: {final_metrics['macro_recall']*100:.1f}%")
    print(f"  * Global Model Macro-F1               : {final_metrics['macro_f1']:.4f}")
    print(f"  * Global Model PR-AUC                  : {final_metrics['pr_auc']:.4f}")
    print("=" * 70)

    # 3. Federated Learning with FedProx (Drift control)
    print("\n[+] Running Federated Learning with FedProx (mu=0.01)...")
    sim_prox = FederatedSimulation(
        num_orgs=5,
        alpha=0.5,
        held_out_attack="infiltration",
        held_out_org=0,
        seed=42,
    )
    sim_prox.prepare_data(num_samples=15000)
    prox_results = sim_prox.run_federated_rounds(num_rounds=8, local_epochs=2, mu_fedprox=0.01)
    prox_final = prox_results["history"][-1]["held_out_metrics"]
    print(f"  * FedProx Global Model Recall          : {prox_final['macro_recall']*100:.1f}%")
    print(f"  * FedProx Global Model Macro-F1        : {prox_final['macro_f1']:.4f}")

    print("\n[SUCCESS] Phase 2 Experiment Complete!")


if __name__ == "__main__":
    run_experiment()
