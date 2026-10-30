# Benchmarks

Reproduce (numpy only, no torch):
```bash
python -m fedthreat.bench.runner configs/bench/headline.yaml        # 80 runs
python -m fedthreat.bench.runner configs/bench/dp_sweep.yaml        # 140 runs   (dp_sweep_quick.yaml: 28)
python -m fedthreat.bench.runner configs/bench/defense_matrix.yaml  # 504 runs   (defense_matrix_quick.yaml: 66)
```
Outputs: `runs/<name>/{results.csv,summary.csv,config.yaml,*.png}`. Committed results in `results/`.

**Read this first.** Data is the *synthetic* flow generator and the model is softmax regression, so these numbers
validate trade-offs and machinery, not real-world detection quality. `defense_matrix_quick` uses 1 seed and
`dp_sweep_quick` 2 seeds (the full configs use 3 and 5). Metrics: PR-AUC and recall at 1% FPR; "held-out" =
recall on an attack family (`infiltration`) that only org 0 ever sees.

## 1. Does federation help with an attack only one org has seen? (headline, 5 seeds, 9 orgs, 40 rounds)
| method | alpha=0.1 held-out recall | alpha=1.0 held-out recall | alpha=1.0 PR-AUC |
|---|---|---|---|
| local-only (orgs that never saw it) | 0.10 | 0.005 | 0.932 |
| FedAvg | 0.20 (high variance) | 0.17 | 0.961 |
| secure sum (1 cohort) | 0.29 | 0.19 | 0.976 |
| cohort median / trimmed | 0.24 | **0.011** | 0.942 |
| coordinate median, Krum | 0.19 / 0.41 | 0.005 / 0.002 | 0.935 / 0.931 |
| centralized (upper bound) | 0.75 | 0.75 | 0.993 |

Takeaway: averaging-style aggregation transfers some knowledge of the rare family (0.17-0.19 vs ~0 locally), but
**median-type robust aggregators discard it**: the one org with the rare attack looks like an outlier. Robustness and
sharing of rare attacks pull in opposite directions; a bigger gap to centralized remains. Variance at alpha=0.1 is
large (whether org 0's data survives depends on cohort assignment).

## 2. Privacy / utility (dp_sweep_quick, 15 orgs, 40 rounds, org-level eps at delta=1e-5)
| sigma | eps | secure_sum PR-AUC / recall@1% | cohort_trimmed PR-AUC / recall@1% |
|---|---|---|---|
| 0 (no DP) | inf | 0.970 / 0.910 | 0.939 / 0.900 |
| 2 | 19.8 | 0.953 / 0.881 | 0.930 / 0.880 |
| 4 | 8.1 | 0.931 / 0.831 | 0.906 / 0.789 |
| 8 | 3.6 | 0.900 / 0.729 | 0.767 / 0.224 |
| 16 | 1.7 | 0.796 / 0.270 | 0.357 / 0.000 |

eps ~ 8 costs a few points of PR-AUC and ~8 points of recall with one big cohort; smaller cohorts (more noise per
participant, since each client adds sigma*C/sqrt(honest_min)) degrade sooner. Below eps ~ 4 detection at a low FPR collapses.
Low-noise rows show eps >> 8: a meaningful guarantee needs large sigma and few rounds.

## 3. Robustness (defense_matrix_quick, 15 orgs, PR-AUC, attacker scale 20)
2 of 15 malicious (inside the cohort design's guarantee: 5 cohorts, median tolerates <= 2 contaminated):

| aggregator | sign_flip | gaussian | label_flip | backdoor | LIE | backdoor ASR |
|---|---|---|---|---|---|---|
| fedavg | 0.150 | 0.096 | 0.952 | 0.807 | 0.949 | 1.00 |
| fedavg + server clip* | 0.952 | 0.938 | 0.952 | 0.922 | 0.949 | 0.14 |
| median* | 0.936 | 0.932 | 0.939 | 0.928 | 0.933 | 0.08 |
| krum* | 0.927 | 0.927 | 0.927 | 0.927 | 0.927 | 0.09 |
| **cohort_median** (private) | 0.976 | 0.939 | 0.938 | 0.929 | 0.948 | 0.10 |
| cohort_trimmed (private) | 0.974 | 0.949 | 0.956 | 0.919 | 0.949 | 0.44 |

4 of 15 malicious (outside the guarantee):

| aggregator | sign_flip | gaussian | label_flip | backdoor | LIE | backdoor ASR |
|---|---|---|---|---|---|---|
| median* | 0.936 | 0.931 | 0.941 | 0.925 | 0.934 | 0.10 |
| **cohort_median** | **0.150** | 0.950 | 0.948 | 0.909 | 0.957 | **0.78** |
| cohort_trimmed | **0.150** | 0.956 | 0.967 | 0.911 | 0.959 | 0.68 |

\* these see every individual update, so they give **no secure-aggregation privacy**.

Takeaways: (a) inside f < cohorts/2 the private cohort pipeline matches non-private robust baselines, and
`cohort_median` is the better default than `cohort_trimmed`; (b) it fails abruptly past that point (4 attackers can
contaminate 4 of 5 cohorts), while plain median/Krum degrade gracefully because they work per client. Privacy via cohorts
buys less robustness per attacker: size cohorts so that cohorts >= 2f + 1 for the f you must tolerate (more orgs, or smaller cohorts at
the cost of anonymity); (c) LIE barely hurt anything here (the linear model on this data is easy), so do not read that as safety:
rerun on your real data/model.

## Known limits of this harness
Synthetic data; linear model; update-level attacks are heuristic (an attacker optimising directly against a known
aggregator is stronger); one backdoor trigger; the cohort path skips the cryptography (verified equivalent in
`tests/test_bench.py`). The DP epsilon assumes the stated number of honest noising clients per cohort.
