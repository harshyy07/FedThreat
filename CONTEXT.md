# CONTEXT.md: Federated Learning Framework for Cross-Organization Threat Detection

> Purpose of this file: single source of truth for building the project. Give it to any engineer or AI coding assistant at the start of a session. Update it as decisions change.

---

## 1. Project Summary

**Goal:** Let multiple organizations (hospitals, banks, universities, etc.) jointly train a threat-detection model without any of them sharing raw logs. Each org trains locally. Only model updates leave the org, and those updates are protected by secure aggregation and differential privacy. The aggregation server is hardened against malicious participants (model poisoning).

**Core claims the system must make true:**
1. **Data locality:** raw logs never leave an org's trust boundary.
2. **Update privacy:** the server (and other orgs) cannot recover individual updates or training records. Protected by secure aggregation plus DP.
3. **Robustness:** up to *f* of *n* malicious orgs cannot meaningfully degrade or backdoor the global model.
4. **Heterogeneity tolerance:** orgs with different log formats, traffic profiles, and attack mixes still benefit from the shared model.
5. **Measurable value:** the federated model beats each org's local-only model on attacks that org has rarely or never seen.

**Non-goals (v1):** real-time inference serving at scale, a polished UI, legal/contractual frameworks for consortium governance, vertical FL (different features for the same entities).

---

## 2. Threat Model (write this down first, everything else follows)

| Adversary | Capability | Defense |
|---|---|---|
| **Honest-but-curious server** | Sees everything the server receives; follows protocol | Secure aggregation (server sees only the sum); DP noise |
| **Malicious server** | May deviate: drop clients, forge aggregates, equivocate | Out of scope for v1 beyond signed/verifiable round transcripts; document as limitation |
| **Malicious client org (Byzantine)** | Sends arbitrary updates; may collude with up to *f* others | Norm clipping, robust aggregation, anomaly scoring, reputation |
| **Data poisoning client** | Flips labels, injects crafted samples locally | Same as above plus local data-quality checks |
| **Backdoor client** | Model-replacement/scaling attacks to embed a trigger ("this C2 pattern = benign") | Clipping, FLTrust-style root dataset, backdoor probes in evaluation |
| **Sybil attacker** | Registers many fake orgs | Authenticated enrollment (mTLS certs issued by a consortium CA), one identity per org |
| **Inference attacker** (any participant) | Membership inference, gradient inversion on updates or on the final model | DP-SGD, secure aggregation, minimum cohort size |
| **Network attacker** | MITM, replay | mTLS, per-round nonces, signed messages |

**Assumptions to state explicitly:**
- Fraction of malicious clients *f/n < 0.3* (Byzantine-robust aggregators typically need less than 1/2; Krum needs *n ≥ 2f + 3*, Bulyan needs *n ≥ 4f + 3*).
- At least *k* clients participate each round (a minimum cohort for privacy).
- The consortium CA is trusted.

**Key tension to keep in mind:** secure aggregation hides individual updates, but robust aggregation (Krum, median, trimmed mean) needs to *inspect* individual updates. See Section 7 for how to resolve this honestly.

---

## 3. Architecture

```
 ┌──────────────────────── Org A (trust boundary) ────────────────────────┐
 │  Raw logs ─► Adapter A ─► Canonical features ─► Local trainer (DP-SGD)│
 │  (never leaves)                                  │                     │
 │                                      clip + noise + mask                │
 └──────────────────────────────────────────────────┼─────────────────────┘
                                                    │ masked update (mTLS)
 ┌──────────────────────── Org B ───────────────────┼─────────────────────┐
 │  Raw logs ─► Adapter B ─► Canonical features ─► Local trainer          │
 └──────────────────────────────────────────────────┼─────────────────────┘
                                                    ▼
                          ┌───────────────────────────────────────────┐
                          │ Aggregation Server                        │
                          │  • Enrollment/auth (mTLS)                 │
                          │  • Round orchestration                    │
                          │  • Secure-aggregation coordinator         │
                          │  • Robust aggregator + update validation  │
                          │  • Privacy accountant (global ε budget)   │
                          │  • Model registry + audit log             │
                          └───────────────────────────────────────────┘
                                                    │ new global model
                                                    ▼  (broadcast to all orgs)
```

### Components

1. **Client (org-side agent):** log adapter, feature extractor, local trainer, DP module, secure-agg client, update signer. Runs entirely in the org's environment (container or VM).
2. **Server:** FastAPI/gRPC service. Receives only masked, clipped, noised updates. Never receives raw data or feature vectors.
3. **Adapter layer:** per-org plugins mapping local log formats to the canonical schema (Section 5).
4. **Simulation harness:** runs N virtual orgs in one process for development and experiments (use this first; deploy later).
5. **Evaluation suite:** attack simulators, privacy attacks, metrics.

### Round protocol (one training round)

1. Server selects participating clients (min cohort *k*), sends global model *w_t* and round config (clip norm *C*, noise multiplier σ, local epochs).
2. Each client trains locally for E epochs on its data (DP-SGD or non-DP plus update-level noise).
3. Client computes Δ = w_local − w_t, **clips** ‖Δ‖₂ ≤ C, optionally adds Gaussian noise.
4. Client performs secure-aggregation handshake (key agreement, mask generation, secret-share keys for dropout recovery), sends masked Δ.
5. Server aggregates (see Section 7), updates w_{t+1}, records privacy spend, logs round metadata.
6. Server broadcasts w_{t+1}. Clients evaluate on a local validation set and report **only aggregate metrics** (optionally with DP noise).

---

## 4. Technology Choices

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| FL framework | **Flower (flwr)** | Easiest path to a custom strategy and simulation; alternatives: FedML, NVIDIA FLARE, TensorFlow Federated |
| ML | PyTorch | |
| DP | **Opacus** (DP-SGD, RDP/PRV accountant) | Alternative: `dp-accounting` from Google |
| Secure agg | Own implementation of Bonawitz et al. pairwise masking (educational/prototype), `cryptography` lib for X25519, HKDF, AES-GCM, Ed25519 | Do **not** invent primitives. For production consider MP-SPDZ, CrypTen, or OpenMined PySyft/TenSEAL (CKKS) |
| Transport | gRPC + mTLS | |
| Config | Hydra or Pydantic settings + YAML | |
| Tracking | MLflow or Weights & Biases (server-side metrics only) | |
| Testing | pytest, hypothesis (for crypto/protocol invariants) | |
| Packaging | Docker, docker-compose for local multi-org demo | |

---

## 5. Handling Heterogeneous Data

Two distinct problems: **schema heterogeneity** (different log formats) and **statistical heterogeneity** (non-IID distributions).

### 5.1 Schema heterogeneity

**Approach: canonical feature space, local adapters.** Never try to federate over raw field names.

- Define a **canonical event schema** based on an open standard (OCSF or Elastic Common Schema) for the fields you care about: timestamp, src/dst IP (hashed or bucketed), ports, protocol, bytes in/out, duration, auth result, process name, user type, etc.
- Each org writes an **adapter** (`adapters/<org>.py`) that converts native logs (Zeek, Suricata, Windows Event, Syslog, Cisco, cloud audit logs, EHR access logs) → canonical events.
- A shared **feature extractor spec** turns canonical events into a **fixed-length feature vector** over time windows (e.g., 5-minute windows per host/flow): counts, rates, entropy of ports/destinations, failed-auth ratio, byte ratios, inter-arrival statistics, categorical features via **feature hashing** (so there's no shared vocabulary that could leak).
- Missing features: use a **mask vector** (1 if feature available) concatenated with zero-imputed features, so the model learns to handle absent fields.
- Versioned feature spec (`feature_spec_v1.yaml`). The server rejects updates whose `feature_spec_hash` does not match.

**Fallback for orgs with very different feature coverage:** a shared encoder plus org-specific input projection layers (partial model sharing: only the shared body is aggregated).

### 5.2 Statistical heterogeneity (non-IID)

Orgs differ in label distribution (one sees mostly phishing, another ransomware), volume, and benign baselines. Techniques to implement and compare:

| Technique | Idea |
|---|---|
| **FedAvg** | Baseline |
| **FedProx** | Proximal term μ‖w − w_t‖² to limit client drift |
| **SCAFFOLD** | Control variates to correct drift (note: doubles communication, interacts awkwardly with DP/robust agg) |
| **FedBN / local normalization** | Keep BatchNorm/LayerNorm statistics local |
| **Personalization** | Fine-tune the global model locally; or shared body + local head |
| **Class-imbalance handling** | Focal loss, class-weighted loss, local re-sampling (all client-side) |
| **Clustered FL** (stretch) | Group orgs with similar distributions |

### 5.3 Model (v1)

- Start with an **MLP** (3 layers, 128-64-32) on windowed features: fast, DP-friendly, easy to debug.
- v2: 1D-CNN or small Transformer over event sequences; autoencoder branch for anomaly detection of unseen attacks.
- Avoid BatchNorm with DP-SGD (use GroupNorm/LayerNorm).
- Output: multi-class (benign + attack families) or binary plus anomaly score.

---

## 6. Differential Privacy

### 6.1 Choose the granularity first
- **Record-level DP** (one log window/flow) is the easiest to achieve and what DP-SGD gives you.
- **Org-level DP** (hide the participation or data of an entire org) is what matters for cross-org sharing and is achieved by clipping the *whole client update* and adding noise to the aggregate. Mostly impractical with only 5-20 orgs because noise dominates; report honestly.
- Decide which you claim. Document it in the README.

### 6.2 Mechanisms to implement

1. **Local DP-SGD (Opacus):** per-sample gradient clipping (C_g), Gaussian noise σ, Poisson sampling rate q; RDP/PRV accountant tracks (ε, δ). Set δ < 1/|D_local|.
2. **Update-level clipping:** clip ‖Δ‖₂ ≤ C before masking. This also bounds the sensitivity for server-side noise and limits poisoning impact.
3. **Distributed DP with secure aggregation:** each client adds noise N(0, σ²C²/n) so the *sum* has total noise σ²C², and the server only sees the sum. This gives central-DP-like utility without trusting the server. Use discrete Gaussian or skellam noise if using integer secure aggregation (modular arithmetic).
4. **Central DP fallback:** server adds noise after aggregation (requires trusting the server; simplest baseline).
5. **Privacy accountant service:** server tracks cumulative ε across rounds; stops training or warns when budget (e.g., ε = 8, δ = 1e-5) is exhausted.

### 6.3 Practical guidance
- Expect an accuracy drop. Run an **ε sweep** (∞, 16, 8, 4, 2, 1) and plot utility vs ε.
- Large batch sizes and fewer rounds improve the privacy/utility trade-off.
- Tune clip norms from observed gradient norm quantiles (median is a good start).
- DP guarantees apply to **parameters released**, so also consider DP on any metrics clients report.

---

## 7. Secure Aggregation and Byzantine Robustness

### 7.1 Secure aggregation (privacy of individual updates)

Implement **pairwise-masking secure aggregation** (Bonawitz et al., 2017):
1. Each client *i* generates an X25519 keypair; server relays public keys (signed).
2. For each pair (i, j), derive shared seed s_ij via ECDH + HKDF; expand with a PRG into mask vectors. Client i adds +mask for j > i and −mask for j < i. Masks cancel in the sum.
3. Each client adds a **self-mask** b_i and Shamir-secret-shares both b_i and its private DH key among others (threshold *t*) to handle dropouts.
4. Server collects masked updates, asks survivors for shares to remove dropped clients' pairwise masks and survivors' self-masks, recovers the sum.
5. Quantize updates to fixed-point integers and operate modulo 2^b.

Test invariants (hypothesis): sum of masked updates minus masks equals the plain sum exactly; protocol survives up to *d* dropouts; server alone learns nothing about a single update (statistical check on masked output).

Alternatives to document or stretch-goal: additive secret sharing across 2-3 non-colluding aggregators (cheap and simple), CKKS homomorphic encryption (TenSEAL), full MPC (MP-SPDZ).

### 7.2 Byzantine-robust aggregation

Implement a pluggable `Aggregator` interface and compare:

| Aggregator | Requirement | Weakness |
|---|---|---|
| FedAvg (baseline) | none | One attacker can dominate |
| **Norm clipping + FedAvg** | server clips | Weak alone, but cheap and always on |
| **Coordinate-wise median** | n > 2f | Statistical inefficiency, hurt by non-IID |
| **Trimmed mean** (β trim) | n > 2f | Needs a β ≥ f/n |
| **Krum / Multi-Krum** | n ≥ 2f + 3 | Fails under strong non-IID; picks a "typical" client |
| **Bulyan** | n ≥ 4f + 3 | Needs many clients |
| **FLTrust** | small clean root dataset at server | Requires root data (can be public/synthetic threat data such as public benchmark traffic) |
| **Cosine-similarity / FoolsGold-style** | none | Good vs sybils, suspicious of legit similar clients |
| **Reputation scoring** | history | Needs rounds to build up, can be gamed slowly |

### 7.3 Resolving the secure-aggregation vs robust-aggregation conflict

Plain secure aggregation reveals only the sum, which is incompatible with per-client inspection. Document these options and implement **at least A and B**:

- **A. Client-side bounding + secure sum (baseline).** Each client clips its update; the server only sees the clipped sum. Limits damage from any one client to C/n per round but does not detect them.
- **B. Cohort-based robust aggregation.** Partition clients into random groups of size *g* (≥ 3). Secure-aggregate within each group; the server then runs median/trimmed mean/Krum over *group sums*. Tolerates attackers as long as fewer than half the groups contain one. Trade-off: smaller anonymity sets.
- **C. Two-server secure computation.** Clients secret-share updates between two non-colluding servers that jointly compute distances/medians via MPC (as in Prio/ELSA-style systems). Higher engineering cost.
- **D. Zero-knowledge range/norm proofs.** Client proves ‖Δ‖ ≤ C without revealing Δ (e.g., Bulletproofs). Prevents the model-replacement scaling attack. Stretch goal.
- **E. Trusted execution environment** (SGX/SEV) aggregator running robust aggregation on plaintext updates inside an enclave. Note the side-channel history of TEEs.

State clearly in the writeup which combination was used and the guarantee level.

### 7.4 Attack simulations (required for the evaluation)

Implement as pluggable `Attacker` clients:
- **Label flipping** (attack → benign)
- **Gaussian / random noise updates**
- **Sign-flipping / scaled negation**
- **Model replacement** (boost malicious update by n/η to override global model)
- **Backdoor / trigger attack** (e.g., a specific port+byte pattern is labeled benign; measure attack success rate, ASR)
- **Little-is-enough / inner-product manipulation** (stays within the benign variance and evades distance-based defenses)
- **Sybil collusion** (multiple identical malicious updates)
- **Adaptive attacker** who knows the aggregator and optimizes against it (the strongest test)

---

## 8. Repository Layout

```
fedthreat/
├── CONTEXT.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── experiment/            # hydra experiment configs (iid, non_iid, attack_x, dp_eps_y)
│   ├── feature_spec_v1.yaml
│   └── aggregator/*.yaml
├── src/fedthreat/
│   ├── adapters/              # base.py, zeek.py, windows_evtx.py, ehr_audit.py, ...
│   ├── features/              # extractor.py, hashing.py, spec.py
│   ├── models/                # mlp.py, cnn1d.py, autoencoder.py
│   ├── client/                # trainer.py, dp.py, secagg_client.py, update_signer.py
│   ├── server/                # app.py, orchestrator.py, enrollment.py, accountant.py, registry.py
│   ├── aggregation/           # base.py, fedavg.py, clip.py, median.py, trimmed_mean.py,
│   │                          # krum.py, bulyan.py, fltrust.py, cohort.py, reputation.py
│   ├── secagg/                # keys.py, masking.py, shamir.py, protocol.py, quantize.py
│   ├── attacks/               # label_flip.py, replacement.py, backdoor.py, lie.py, adaptive.py
│   ├── privacy_eval/          # membership_inference.py, gradient_inversion.py
│   └── sim/                   # partitioner.py, run_simulation.py
├── data/                      # (gitignored) datasets and partitions
├── tests/
│   ├── unit/ , integration/ , crypto_properties/
├── scripts/                   # download_data.sh, make_partitions.py, plot_results.py
├── docker/                    # client.Dockerfile, server.Dockerfile, compose.yaml
└── docs/                      # threat_model.md, protocol.md, results.md
```

---

## 9. Data and Evaluation Plan

### 9.1 Datasets
- **CIC-IDS2017 / CSE-CIC-IDS2018** (network flows, many attack families), **UNSW-NB15**, **NSL-KDD** (smoke tests only; outdated), **CIC-DDoS2019**, **BETH** (host-level), **LANL authentication logs** (auth anomaly), **Bot-IoT/ToN-IoT** (IoT).
- Verify each dataset's license/terms before use.

### 9.2 Simulating organizations (the partitioner)
Create 5-10 virtual orgs with:
- **Label skew:** Dirichlet(α) allocation of attack classes (α = 0.1 very non-IID, 1.0 mild, 100 ≈ IID).
- **Quantity skew:** log-normal org sizes.
- **Schema skew:** drop or rename fields per org via adapters (e.g., org C lacks DNS data).
- **Temporal skew:** orgs see different time slices.
- **Held-out attack:** at least one attack family is seen only by one org; test whether others detect it after federation. This is the headline experiment.

### 9.3 Baselines (always compare against these)
1. **Local-only** per org.
2. **Centralized** (pooled data; the upper bound).
3. **Naive FedAvg** with no privacy or defenses.

### 9.4 Metrics
- Detection: per-class precision/recall/F1, macro-F1, **PR-AUC** (data is highly imbalanced; do not rely on accuracy), **recall at fixed FPR (e.g., 0.1%)** since SOC alert fatigue is real.
- Robustness: accuracy drop under attack vs. fraction of malicious clients; **backdoor ASR**; detection rate of malicious clients (if the aggregator flags them).
- Privacy: (ε, δ); empirical membership-inference AUC (target ≈ 0.5); gradient-inversion reconstruction quality (on non-secure-agg baseline to show why protection is needed).
- Systems: rounds to convergence, communication bytes per round, client compute time, secure-agg overhead, dropout tolerance.

### 9.5 Experiment matrix (minimum)
- Aggregators × {IID, mild non-IID, severe non-IID} × {no attack, each attack at 10/20/30%} 
- DP ε sweep × utility
- Secure agg on/off × overhead
- Ablation: each defense layer added one at a time

---

## 10. Phased Roadmap with Acceptance Criteria

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Foundations** | Repo, threat model doc, canonical schema, feature spec, dataset download scripts | `docs/threat_model.md` reviewed; features reproducible from a fixed seed |
| **1. Centralized baseline** | MLP trained on pooled data; metrics pipeline | Reproduces reasonable macro-F1 and PR-AUC on chosen dataset |
| **2. Federated simulation** | Flower simulation, partitioner, FedAvg/FedProx | Federated beats local-only on held-out-attack experiment; approaches centralized |
| **3. Heterogeneity** | Adapters for 3 log formats, mask vector, partial sharing | Same model trains across orgs with different available fields; per-schema test coverage |
| **4. Differential privacy** | Opacus integration, accountant, ε sweep | Utility-vs-ε curve; accountant stops at budget; MIA AUC reported |
| **5. Robust aggregation** | Krum, trimmed mean, median, clipping, FLTrust, cohort | All attacks in 7.4 implemented; defense matrix results; documented failure cases |
| **6. Secure aggregation** | Pairwise-masking protocol with dropout recovery | Property tests pass; exact-sum equality; handles ≥ 20% dropout |
| **7. Integration** | DP + secure agg + robust cohorts together | End-to-end round works; overhead measured; guarantees written precisely |
| **8. Real deployment demo** | Docker-compose with 3 client containers and 1 server over mTLS | One command brings up the demo; no raw data crosses containers (verified by network capture) |
| **9. Writeup** | `docs/results.md`, limitations, future work | Honest limitations section |

---

## 11. Security Engineering Requirements

- **Identity:** mTLS with a consortium CA; one certificate per org; revocation list; per-round signed messages and nonces (anti-replay).
- **Update validation (server):** schema/shape/dtype check; finite-values check (no NaN/Inf); norm bound; feature-spec hash; round ID match.
- **Minimum cohort:** server refuses to aggregate with fewer than *k* clients (privacy and anonymity).
- **Audit log:** append-only, hash-chained log of rounds, participants, aggregator decisions, ε spend.
- **Model integrity:** sign released global models; clients verify before loading (avoid server-pushed backdoored models if the server is compromised; at least detect inconsistency).
- **Secrets:** never log updates, keys, or feature vectors. Redact in debug logs.
- **Supply chain:** pin dependencies, hash-verified installs.
- **Data minimization on clients:** no raw log, IP, or user identifier leaves the client; if IPs are used as features, hash with an org-local secret salt.
- **Compliance notes:** model updates can still count as personal data under GDPR if they allow re-identification; HIPAA environments need local deployment inside the covered entity. Involve counsel before any real deployment. This project is a research/engineering framework, not legal advice.

---

## 12. Known Pitfalls (read before debugging)

1. **DP-SGD + BatchNorm** breaks per-sample gradients. Use GroupNorm/LayerNorm.
2. **Secure agg needs integers.** Fixed-point quantization range must be tuned (clip norm drives it). Overflow produces silent garbage.
3. **Krum under non-IID** often discards honest but unusual clients, which are exactly the ones with rare attacks, so you lose the benefit of federation. Always check this.
4. **Class imbalance** makes accuracy meaningless; a model that predicts "benign" scores 99%+.
5. **Noise scale vs. few clients:** with ~5 orgs, org-level DP noise swamps the signal. Be honest; report record-level DP primarily.
6. **Backdoors survive median/Krum** if the attacker stays near the benign cluster (LIE attack). Always test adaptive attackers.
7. **Poisoning amplification across rounds.** Evaluate over many rounds, not one.
8. **Metric leakage:** reporting per-client validation metrics can leak information. Aggregate them or add noise.
9. **Dataset leakage in simulation:** time-based and flow-based splits; avoid the same flow in train and test.
10. **Reproducibility:** fix seeds for partitioning, model init, DP noise (in tests only), and attacker selection.

---

## 13. Conventions

- Type hints everywhere; `ruff` + `black` + `mypy` in CI.
- Every aggregator, attacker, and adapter implements an abstract base class and is registered in a registry, so experiments are config-driven.
- Every experiment writes a config snapshot, git SHA, seed, and metrics to `runs/<timestamp>/`.
- Crypto code lives only in `secagg/` and `server/enrollment.py`; use vetted libraries for primitives; document every protocol step against the paper.
- Tests required for: sum-correctness of secure agg, dropout recovery, DP accountant monotonicity, each aggregator's behavior on synthetic benign + attacker vectors, adapter→feature determinism.

---

## 14. Key References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx); Karimireddy et al., *SCAFFOLD*
- Bonawitz et al., *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
- Abadi et al., *Deep Learning with Differential Privacy* (DP-SGD); Mironov, *Rényi DP*
- Kairouz et al., *The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation*
- Blanchard et al., *Krum*; Yin et al., *Coordinate-wise Median/Trimmed Mean*; El Mhamdi et al., *Bulyan*
- Cao et al., *FLTrust*; Fung et al., *FoolsGold*
- Bagdasaryan et al., *How To Backdoor Federated Learning*; Baruch et al., *A Little Is Enough*; Fang et al., *Local Model Poisoning Attacks to Byzantine-Robust FL*
- Geiping et al., *Inverting Gradients*; Shokri et al., *Membership Inference Attacks*
- Kairouz et al., *Advances and Open Problems in Federated Learning*

---

## 15. Open Questions (decide and record answers here)

- [ ] Privacy claim: record-level only, or also org-level?
- [ ] Secure-agg + robustness strategy for v1: cohort-based (B) or two-server (C)?
- [ ] Primary dataset and the held-out attack family for the headline experiment
- [ ] Is a small clean root dataset available for FLTrust (public benchmark data)?
- [ ] Target ε and δ
- [ ] Do we support asynchronous or only synchronous rounds in v1?
- [ ] Anomaly detection (unsupervised) in v1 or v2?

---

## 16. Suggested First Session Prompt for an AI Coding Assistant

> "Read CONTEXT.md. Start Phase 0 and 1: scaffold the repo per Section 8, implement the canonical feature spec and extractor with feature hashing, download and partition CIC-IDS2017 into 5 non-IID virtual orgs per Section 9.2, and train a centralized MLP baseline reporting macro-F1, PR-AUC, and recall@0.1% FPR. Add tests and a config-driven experiment runner. Do not implement privacy or aggregation features yet."
