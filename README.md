# FedThreat: Federated, Privacy-Preserving, Byzantine-Robust Threat Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework: Flower](https://img.shields.io/badge/FL_Framework-Flower-orange.svg)](https://flower.ai/)
[![Security: mTLS](https://img.shields.io/badge/Transport-mTLS_X.509-green.svg)]()

FedThreat enables multiple organizations (hospitals, banks, enterprises, SOCs) to collaboratively train a shared cyber threat detection model **without sharing raw event logs**. Local parameter updates are bounded via norm clipping, optionally perturbed with differential privacy (DP), masked using cryptographic pairwise secure aggregation across dynamically assigned cohorts, and aggregated using Byzantine-robust statistical aggregators to defend against data and model poisoning attacks.

---

## Key Highlights & Architectural Guarantees

1. **Data Locality**: Raw security logs (Zeek, Syslog, Windows Event Logs) never cross the organization perimeter. Only bounded model updates leave the local enclave.
2. **Cryptographic Secure Aggregation**: Bonawitz-style $t$-of-$n$ pairwise masking (integers mod $2^{32}$) over ECDH X25519 key agreements with Shamir secret sharing for robust dropout recovery.
3. **Byzantine Fault Tolerance & Poisoning Defense**: Implements Coordinate-wise Median, Trimmed Mean, Krum, Bulyan, and Cohort-based robust aggregation to mitigate model replacement, backdoor injection, label-flipping, Gaussian noise, and *A Little Is Enough* (LIE) attacks.
4. **Differential Privacy (DP)**: Local & distributed differential privacy mechanisms with an integrated Rényi Differential Privacy (RDP) privacy accountant to strictly track cumulative $(\epsilon, \delta)$ privacy budgets.
5. **Schema & Log Heterogeneity**: Native format adapters (Zeek connection logs, Windows EVTX/Sysmon, RFC-5424 Syslog) converting raw records into canonical schema representations with cryptographic feature hashing.
6. **Zero-Trust Transport & Identity**: Mutual TLS (mTLS) with custom consortium X.509 PKI, per-round anti-replay nonces, and cryptographic feature specification fingerprinting.

---

## Repository Structure

```text
fedthreat/
├── configs/
│   ├── bench/                 # Benchmark configurations (headline, dp_sweep, defense_matrix)
│   └── feature_spec_v1.yaml   # Canonical feature specification schema
├── docker/                    # Container definitions & compose orchestrations
│   ├── client.Dockerfile
│   ├── server.Dockerfile
│   └── compose.yaml
├── docs/                      # Architectural specs & security models
│   ├── benchmarks.md          # Benchmark findings, trade-offs, and empirical results
│   └── threat_model.md        # Formal adversarial capabilities and threat assumptions
├── results/                   # Empirical benchmark artifacts, curves, and summaries
│   ├── defense_matrix_quick/  # Byzantine defense ablation across 5 attack families
│   ├── dp_sweep_quick/        # Privacy/utility Pareto frontier sweeps
│   └── headline/              # Held-out zero-day attack transfer evaluation
├── scripts/                   # PKI generators, runners, simulation & sweep scripts
├── src/fedthreat/
│   ├── adapters/              # Format adapters (Zeek, Windows EVTX, Syslog)
│   ├── aggregation/           # Robust statistics (Median, Trimmed Mean, Krum, Bulyan, Cohorts)
│   ├── attacks/               # Byzantine attack simulators (poisoning, backdoors, LIE)
│   ├── bench/                 # Benchmark engine, runner, metrics (PR-AUC, recall@FPR), plots
│   ├── client/                # Org agent, local trainers (PyTorch & NumPy), DP accountant
│   ├── features/              # Feature extractors, schema hashing & spec validation
│   ├── models/                # ThreatMLP and parameter serialization
│   ├── secagg/                # Bonawitz-style secure aggregation protocol
│   └── server/                # FastAPI aggregation server, mTLS coordinator & validators
└── tests/                     # Unit, integration, crypto property, and benchmark tests
```

---

## Quick Start

### 1. Installation

Python 3.11 or later is required:

```bash
# Clone the repository
git clone https://github.com/harshyy07/FedThreat.git
cd FedThreat

# Install FedThreat in editable development mode
pip install -e ".[dev]"

# Optional: Install PyTorch and Flower for neural network simulation
pip install -e ".[torch,fl]"
```

### 2. Run the Test Suite

Execute the full verification suite (68 tests covering crypto, adapters, DP, aggregators, server validation, and benchmarks):

```bash
pytest
```

### 3. Generate Dev PKI & Run Live Demo

Simulate an end-to-end multi-organization federated learning round over authenticated mTLS:

```bash
# Generate root CA, server certificates, and org client keypairs
python scripts/generate_certs.py --out certs

# Compute the feature specification hash
export FEDTHREAT_SPEC_HASH=$(python scripts/print_spec_hash.py certs)

# Execute the live multi-client demo (with Byzantine adversary defended)
python scripts/run_live_demo.py
```

---

## Benchmarks & Empirical Findings

FedThreat provides an automated benchmarking harness (`fedthreat.bench.runner`) evaluating non-IID data skew, differential privacy utility degradation, and Byzantine attack resistance.

### Executing Sweeps

```bash
# 1. Headline experiment: rare/held-out zero-day attack detection across Dirichlet skews
python -m fedthreat.bench.runner configs/bench/headline.yaml

# 2. Privacy vs utility sweep (RDP accountant across epsilons)
python -m fedthreat.bench.runner configs/bench/dp_sweep_quick.yaml

# 3. Byzantine defense matrix (evaluates 5 attack models against all aggregators)
python -m fedthreat.bench.runner configs/bench/defense_matrix_quick.yaml
```

Outputs, metric summaries (`results.csv`, `summary.csv`), and generated curves (`headline.png`, `dp_tradeoff.png`, `attack_pr_auc.png`, `backdoor_asr.png`) are saved under `results/` and `runs/`.

For in-depth analysis, empirical trade-offs, and methodology, see [`docs/benchmarks.md`](docs/benchmarks.md).

---

## Security & Privacy Guidelines

- **Key Management**: Certificates and private keys generated in `certs/` or `private_ca/` must **never** be committed to version control.
- **Model Poisoning Bound**: Cohort-based secure robust aggregation guarantees Byzantine tolerance when $f < \text{cohorts} / 2$. Ensure minimum cohort thresholds are calibrated to expected adversary ratios.
- **Compliance & Privacy**: DP guarantees protect released model parameters; ensure local client datasets comply with applicable regulatory privacy standards (HIPAA, GDPR) before deployment.

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
