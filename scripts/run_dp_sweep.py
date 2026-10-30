"""Phase 4 Differential Privacy Evaluation: Epsilon sweep vs. Utility analysis."""

from fedthreat.sim.run_simulation import FederatedSimulation


def run_dp_sweep():
    print("=" * 70)
    print("FedThreat: Phase 4 - Differential Privacy Sweep & Utility Analysis")
    print("=" * 70)
    print("Target delta: 1e-5 | Sensitivity bound (Clip Norm C): 1.0")
    print("Evaluating utility across privacy regimes (Non-DP vs. Moderate DP vs. Strict DP)...")
    print("=" * 70)

    # Noise multipliers to test: 0.0 (infinity eps), 0.5 (moderate), 1.0 (strong)
    dp_configs = [
        {"name": "No DP (sigma=0.0)", "sigma": 0.0, "clip": None},
        {"name": "Moderate DP (sigma=0.3)", "sigma": 0.3, "clip": 1.0},
        {"name": "Strong DP (sigma=0.8)", "sigma": 0.8, "clip": 1.0},
    ]

    results = []

    for cfg in dp_configs:
        print(f"\n[+] Running {cfg['name']} for 5 rounds...")
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
            clip_norm=cfg["clip"],
            noise_multiplier=cfg["sigma"],
            target_delta=1e-5,
        )

        final_step = out["history"][-1]
        eps = out["final_epsilon"]
        f1 = final_step["held_out_metrics"]["macro_f1"]
        pr_auc = final_step["held_out_metrics"]["pr_auc"]

        results.append({
            "name": cfg["name"],
            "sigma": cfg["sigma"],
            "epsilon": eps if cfg["sigma"] > 0 else float("inf"),
            "macro_f1": f1,
            "pr_auc": pr_auc,
        })

    print("\n" + "=" * 70)
    print(f"{'Configuration':<25} | {'Epsilon (eps)':<14} | {'PR-AUC':<10} | {'Macro-F1':<10}")
    print("-" * 70)
    for r in results:
        eps_str = f"{r['epsilon']:.2f}" if r["epsilon"] != float("inf") else "inf (No DP)"
        print(f"{r['name']:<25} | {eps_str:<14} | {r['pr_auc']:<10.4f} | {r['macro_f1']:<10.4f}")
    print("=" * 70)
    print("[SUCCESS] Phase 4 DP Evaluation Complete!")


if __name__ == "__main__":
    run_dp_sweep()
