# Machine-Learning Network Intrusion Detection on NSL-KDD

This project classifies network connection records as **normal** or **attack**, and
assigns attacks to one of four types (**DoS, Probe, R2L, U2R**). It runs on the real
NSL-KDD benchmark. Everything (data download, preprocessing, training of four models
on two tasks, evaluation, figures) comes from one command. The write-up is in
[`paper/paper.md`](paper/paper.md).

```bash
pip install -r requirements.txt
python run_pipeline.py            # ~7 min on 4 CPU cores; writes results/
```

## Headline results (trained on KDDTrain+, tested on KDDTest+)

| Task | Model | Accuracy | Precision | Recall (DR) | F1 | ROC-AUC | False-alarm rate |
|---|---|---|---|---|---|---|---|
| Binary | Logistic regression | 0.745 | 0.913 | 0.609 | 0.731 | 0.794 | 0.076 |
| Binary | Random forest | 0.775 | 0.968 | 0.625 | 0.759 | **0.962** | 0.027 |
| Binary | Gradient boosting | **0.803** | **0.970** | **0.675** | **0.796** | 0.961 | 0.028 |

| Task | Model | Accuracy | Macro-F1 | Recall R2L | Recall U2R |
|---|---|---|---|---|---|
| 5-class | Gradient boosting | **0.778** | 0.554 | 0.077 | 0.119 |
| 5-class | Gradient boosting, balanced class weights | 0.778 | **0.613** | 0.183 | 0.224 |

On a random 20% hold-out of the *training* file every tree model scores above 99.9%
accuracy. On the official test file the same models drop to 77–80%. That gap is the
central finding of the paper. It comes from distribution shift and from 17 attack
types that appear only in the test set: gradient boosting detects 77% of known-type
attacks but only 44% of novel-type attacks. Full tables are in
[`results/`](results/) and figures in [`results/figures/`](results/figures/).

## Repository layout

```
run_pipeline.py          one-command entry point (python run_pipeline.py [--synthetic])
src/
  config.py              paths, seed, 41-feature schema, attack -> category mapping
  data.py                download + SHA-256 check of NSL-KDD, loading, label derivation
  synthetic.py           schema-identical synthetic generator (offline fallback only)
  preprocessing.py       ColumnTransformer: one-hot / log1p / standard scaling
  models.py              majority baseline, logistic regression, random forest, gradient boosting
  evaluation.py          binary & multi-class metrics (incl. false-alarm rate, macro scores)
  plots.py               all figures (matplotlib, fixed colour per model/class)
  pipeline.py            orchestrates the experiment protocol, writes results/
tests/test_smoke.py      offline unit + end-to-end smoke tests (synthetic data)
results/                 generated metrics CSVs, confusion matrices, figures, run_info.json
paper/paper.md           the research paper
```

## Data: which path was taken

**The real NSL-KDD dataset was used for every reported number.**
`results/run_info.json` records `"data_source": "nsl-kdd"`.

* The canonical host (UNB/CIC) puts the files behind a registration form, so
  `src/data.py` fetches `KDDTrain+.txt` and `KDDTest+.txt` from public GitHub mirrors.
  It rejects any file whose SHA-256 does not match the pinned hash in `src/config.py`.
  Files are cached in `data/raw/`, which git ignores.
* Sizes match the published dataset: 125,973 training records, 22,544 test records,
  and 11,850 in the hard subset KDDTest-21 (`difficulty < 21`).
* **Fallback:** if no mirror is reachable, or you pass `--synthetic`, the pipeline uses
  `src/synthetic.py` instead. It generates a dataset with the same 41-column schema,
  built from per-attack traffic signatures such as SYN floods (flag `S0`, `serror_rate≈1`),
  ICMP floods, port scans and password guessing. Its test split also contains attack
  types that are absent from training. The fallback only keeps the code runnable offline
  and is used by the smoke test. Its numbers are **not** comparable with NSL-KDD results,
  and `run_info.json` labels the run as `synthetic`.

## Threat model

* **Defender / deployment point:** a network-based IDS that sees connection summaries,
  e.g. NetFlow/Zeek-style records enriched with the 2-second and 100-connection window
  statistics that NSL-KDD provides. Content features such as `num_failed_logins` assume
  some host or protocol parsing.
* **Adversary:** an external or internal attacker who sends traffic through the
  monitored link. Attack goals cover the four DARPA-98 categories:
  * **DoS:** make a service unavailable.
  * **Probe:** reconnaissance.
  * **R2L:** gain user access remotely.
  * **U2R:** escalate a local account to root.
* **What the model is meant to catch:** attacks whose *traffic statistics* resemble
  attacks in the training data, including new variants of known families.
* **Out of scope / not defended:**
  * Encrypted payload inspection.
  * Adversaries who deliberately shape traffic to evade the classifier (adversarial ML).
  * Poisoning of training data.
  * Attacks that look statistically identical to normal traffic at the connection level.
  
  Section 4.4 of the paper shows the last case explicitly: `snmpgetattack` and
  `guess_passwd` in KDDTest+ are almost never detected.
* **Operational cost model:** false alarms consume analyst time, and most traffic is
  benign. So we report the **false-alarm rate** next to the detection rate, and macro
  averages so that the rare, high-impact R2L and U2R classes are not hidden by DoS volume.

## Design choices (and why)

| Choice | Rationale |
|---|---|
| Official split: train on KDDTrain+, test on KDDTest+ and KDDTest-21 | A random split of one file overstates real-world performance. The official test file has a different distribution and 17 unseen attack types, which is a realistic proxy for deployment. We report both splits to show the gap. |
| Preprocessing inside an sklearn `Pipeline` | The encoder vocabulary and scaler statistics are fitted only on the training data, so nothing leaks from the test set. |
| One-hot encoding with `handle_unknown="ignore"` | `service` has 70 values, and new services appear in production. An unknown value becomes an all-zero vector instead of causing a crash. |
| `log1p` then standardisation for byte/count features | `src_bytes` spans 0 to 1.4e9. Without the log, a few extreme values dominate the linear model. Trees are scale-invariant, so one shared pipeline keeps the comparison fair. |
| Drop `num_outbound_cmds` | It is constant (always 0) in NSL-KDD. |
| Four models: majority, logistic regression, random forest, gradient boosting | A floor, a linear baseline and two standard tree ensembles. All are in scikit-learn. |
| Fixed, conventional hyper-parameters, no tuning on test | Tuning on KDDTest+ would leak test information. A random validation split cannot rank configurations because everything scores above 99%. |
| Macro-F1, per-class recall and false-alarm rate | Accuracy is dominated by normal and DoS traffic. These metrics expose the R2L/U2R blind spot and the operational cost. |
| Class-weight ablation | Tests whether re-weighting the rare classes helps. It does for boosting and logistic regression, but not for the random forest. |
| Permutation importance on the test set, alongside impurity importance | Impurity importance is biased toward high-cardinality features. Permutation importance on held-out data measures what actually drives generalisation. |
| Global seed 42, deterministic estimators | Re-running reproduces the same numbers. |

## Reproducing and testing

```bash
python run_pipeline.py                 # real data -> results/
python run_pipeline.py --synthetic --out /tmp/synth   # offline fallback
python -m unittest discover -s tests -v                # ~1 min, no network
```

The results were generated with Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9
and matplotlib 3.11 (exact versions are in `results/run_info.json`). Other recent
versions should work. Small numeric differences across library versions are possible.

## Outputs in `results/`

| File | Content |
|---|---|
| `metrics_comparison_binary.csv` | accuracy, precision, recall, F1, ROC-AUC, false-alarm rate for each model and evaluation set |
| `metrics_comparison_multiclass.csv` | accuracy, macro P/R/F1, weighted F1, macro OvR ROC-AUC, per-class recall, plus the class-weight ablation |
| `metrics_all.csv` | every metric for every run, including confusion-matrix counts and fit times |
| `confusion_*.csv` | confusion matrices for each task, model and test set |
| `classification_reports.txt` | per-class sklearn reports on KDDTest+ |
| `novel_vs_known_detection.csv`, `detection_rate_per_attack_type.csv` | detection rate on attack types seen and unseen in training |
| `operating_points_binary.csv` | detection rate at a false-alarm rate of 1/2/5/10% (oracle threshold, a diagnostic of ranking quality) |
| `feature_importance.csv` | impurity and permutation importance for each original feature |
| `dataset_summary.csv`, `attack_type_counts.csv` | class and attack-type counts per split |
| `figures/*.png` | class distribution, confusion matrices, ROC curves (binary and one-vs-rest), feature importance, per-class recall, known-vs-novel detection |
