# FedThreat Threat Model and Verified Security Status

Status legend: **Tested** = an automated test in `tests/` fails if the property breaks.
**Implemented** = code exists but is only exercised indirectly. **Not implemented** = a known gap.
Claims here were re-audited against the code; the previous version overstated several items.

## Trust assumptions
| Party | Assumption |
|---|---|
| Consortium CA / enrollment registry | Trusted and correct. |
| Aggregation server | **Honest-but-curious** for secure aggregation. A server that lies about the survivor set can extract more than the sum (needs Bonawitz's consistency round: **not implemented**). It is *not* trusted with plaintext updates. |
| Orgs | Up to *f* may be Byzantine, with f < (number of cohorts)/2 and fewer than a Shamir threshold colluding with the server inside one cohort. |

## Properties
| Property | Mechanism | Status |
|---|---|---|
| Raw data never leaves an org | Agent only sends public keys, ciphertexts, masked uint32 vectors, key shares | Tested (`test_no_plaintext_update_appears_on_the_wire`) |
| Server sees only cohort **sums** | Client-side pairwise + self masking over Z_2^32; PRG = SHAKE-256 on full 32-byte seeds, domain-separated per tensor/round | Tested (`test_secagg.py`, `test_server_e2e.py`) |
| Dropout tolerance | Shamir sharing of mask key (dropped) / self seed (survivor); client refuses to reveal both for one peer | Tested: before-masking and after-masking dropouts, end to end |
| Authenticated key exchange | Ed25519-signed advertisements; clients detect server key substitution; AES-GCM share transport | Tested |
| Client authentication | mTLS with **required** client certs (`--ssl-cert-reqs 2`) + per-request Ed25519 signature, timestamp window, nonce replay cache; identity taken from the signature, never from a payload field | Tested at app level; TLS layer exercised by `scripts/run_live_demo.py` (cert-less client refused). Docker path **not** exercised by tests. |
| Sybil resistance | One identity per enrolled org; duplicate submissions rejected | Tested (duplicate advertise); depends on enrollment governance |
| Feature-spec consistency | Server rejects mismatched `spec_hash`; clients refuse mismatched servers | Tested |
| Byzantine robustness | Random cohorts (>= 3 required), cohort-sum norm clipping, median/trimmed-mean/Krum across cohort means | Tested (attacker at 3 positions; single-cohort mode is flagged non-robust; rounds **abort** instead of silently losing robustness) |
| DoS by one attacker | Anomalous cohort sums are **clipped**, not dropped, so one org cannot push valid cohorts below the minimum | Tested |
| Org-level (eps, delta)-DP of released cohort sums | Client clipping (C) + distributed Gaussian noise (sigma*C/sqrt(h)); RDP accountant with pre-flight budget check; training halts when exhausted | Accountant **Tested** vs Google `dp-accounting`; halting Tested. The guarantee needs >= `honest_min` honest, noising members per cohort. DP is **off by default** and reported as `epsilon: null`, never a fake number. |
| Membership-inference resistance | None measured | **Not implemented** (no attack evaluation yet) |

## Known limitations (read before deploying)
1. **Utility cost of org-level DP is large.** Noise norm is about sigma*C*sqrt(d). With d ~ 22k parameters and few orgs the signal is swamped unless sigma is small; report the eps-vs-utility curve before claiming privacy.
2. **Cohort anonymity is small** (3 orgs). Larger cohorts give more privacy and less robustness; f attackers need >= 2f+1 cohorts (>= 9 orgs for f = 1).
3. Norm clipping by clients is **not verifiable** by the server (the server only sees sums). The cohort-sum clip bounds the damage but a Byzantine org can still push its own cohort to the bound. ZK range proofs are future work.
4. Two attackers in different cohorts can contaminate 2 of 3 cohorts and defeat the median (`test_two_attackers_in_three_cohorts_can_break_median`).
5. Noise is float64 Gaussian rounded by 2^-16 quantisation; a discrete Gaussian sampler is the production-grade choice.
6. A client that drops *after* masking only costs a missing revealer; if >= (cohort - threshold + 1) members of one 3-cohort vanish the cohort fails and the round may abort.
7. Torch trainer (`ClientTrainer`) delegates clip/noise to the tested numpy helpers, but the torch path was **not executed** in the audit environment.
8. Self-reported `num_samples` are ignored on purpose: each org counts equally.
9. Server TLS private key and registry live on the server host; use an HSM/secret manager in production. The dev PKI in `scripts/generate_certs.py` is for testing only.
