# Model Eval

Experimental settings for Section 6.2, **Characterizing Models with Less Evaluation**. Both studies use evaluation data from **18 benchmarks** and normalized scores in [0,1], including partial credit.

[Tech Report](https://arxiv.org/abs/2610.24739) · [Experimental results](experimental-results.md#model-eval-characterizing-models-with-less-evaluation)

These studies use separate model and query inventories from RobustRouteBench. The **Graph** predictor is distinct from the **GraphRouter** routing baseline. Results appear in report §6.2; predictor implementations and full response tables are not released.

## 1. Experimental settings

| Setting | Figures 10–11: within-domain estimation | Figure 12: cross-domain prediction |
|---|---|---|
| Models / queries | 25 / 25,504 | 35 / 25,457 |
| Historical / evaluation models | Fixed temporal split: 18 / 7 | Fixed split: 30 / 5 |
| Prediction groups | 18 benchmarks | 32 behavioral domains |
| Historical evidence | 1% and 5% visible model–query scores | 20%, 40%, 60%, and 80% missing model–domain scores |
| Per-model evaluation budget | 5, 10, 15, 20, 25, 30, 35, 40 queries across the pool | 0, 1, 2, 4, 8 measured domains |
| Repetitions | 5 historical runs × 7 evaluation models × 3 query seeds | 3 fitting seeds × 5 evaluation models × 3 selection seeds |
| Metric | MAE over all 18 benchmark means | MAE over unmeasured domains |

### Data and model inventories

| Benchmark | Queries: Figures 10–11 | Queries: Figure 12 |
|---|---:|---:|
| `BLINK` | 1,901 | 1,900 |
| `CharXiv_reasoning_val` | 1,000 | 998 |
| `ChartQAPro` | 1,948 | 1,948 |
| `ERQA` | 400 | 399 |
| `HallusionBench` | 951 | 948 |
| `InfoVQA_VAL` | 2,801 | 2,791 |
| `LogicVista` | 447 | 447 |
| `MMMU_Pro_10c` | 1,730 | 1,730 |
| `MMStar` | 1,500 | 1,485 |
| `MathVerse_MINI_Vision_Only` | 788 | 784 |
| `MathVision` | 3,040 | 3,040 |
| `MathVista_MINI` | 1,000 | 998 |
| `OCRBench` | 1,000 | 999 |
| `RealWorldQA` | 765 | 764 |
| `SEEDBench2_Plus` | 2,277 | 2,275 |
| `SimpleVQA` | 2,025 | 2,022 |
| `VStarBench` | 191 | 191 |
| `WeMath` | 1,740 | 1,738 |
| **Total** | **25,504** | **25,457** |

The query-level split uses release-date boundaries of **21 October 2025** for the 18/7 historical–evaluation split and **26 August 2025** for the internal 12/6 fitting–development split. Equal-date release batches stay together. Final fitting uses all 18 historical models.

<details>
<summary>Model inventories and query-level release dates</summary>

Historical models are listed in the final encoder’s reference-coordinate order.

| Historical model | Release date / public-by bound | Role during internal development |
|---|---|---|
| `InternVL3-2B-Instruct` | Public by 2025-04-17 | Internal fitting |
| `InternVL3_5-14B-Instruct` | 2025-08-26 | Development |
| `InternVL3_5-2B-Instruct` | 2025-08-26 | Development |
| `InternVL3_5-4B-Instruct` | 2025-08-26 | Development |
| `InternVL3_5-8B-Instruct` | 2025-08-26 | Development |
| `InternVL3_5-GPT-OSS-20B-A4B-Preview` | 2025-08-26 | Development |
| `MiMo-VL-7B-RL` | Public by 2025-05-30 | Internal fitting |
| `Ovis2-1B` | 2025-01-26 | Internal fitting |
| `Ovis2-2B` | 2025-01-26 | Internal fitting |
| `Ovis2-4B` | 2025-01-26 | Internal fitting |
| `Ovis2-8B` | 2025-01-26 | Internal fitting |
| `Ovis2.5-2B` | 2025-08-15 | Internal fitting |
| `Ovis2.5-9B` | 2025-08-15 | Internal fitting |
| `Qwen2.5-VL-32B-Instruct` | 2025-03-24 | Internal fitting |
| `Qwen2.5-VL-3B-Instruct` | 2025-01-28 | Internal fitting |
| `Qwen2.5-VL-72B-Instruct` | 2025-01-28 | Internal fitting |
| `Qwen2.5-VL-7B-Instruct` | 2025-01-28 | Internal fitting |
| `Qwen3-VL-30B-A3B-Instruct` | 2025-10-04 | Development |

| Final evaluation model | Release date |
|---|---|
| `Qwen3-VL-32B-Instruct` | 2025-10-21 |
| `Qwen3.5-27B` | 2026-02-24 |
| `Qwen3.5-9B` | 2026-03-02 |
| `Qwen3.6-35B-A3B` | 2026-04-16 |
| `Qwen3.6-27B` | 2026-04-22 |
| `Keye-VL-2.0-30B-A3B` | 2026-05-25 |
| `Gemma4-12B-it` | 2026-06-03 |

“Public by” denotes an upper bound on the release date. Five of the seven evaluation models belong to the Qwen family.

Figure 12 uses the following 35-model inventory, split into 30 historical and five evaluation models:

| Model in the domain-level inventory |
|---|
| `InternVL3-2B-Instruct` |
| `InternVL3_5-2B-Instruct` |
| `InternVL3_5-4B-Instruct` |
| `Ovis2-1B` |
| `Ovis2-2B` |
| `Ovis2-4B` |
| `MiMo-VL-7B-RL` |
| `InternVL3-8B-Instruct` |
| `Ovis2-8B` |
| `InternVL3_5-14B-Instruct` |
| `InternVL3_5-8B-Instruct` |
| `Keye-VL-2.0-30B-A3B` |
| `Ovis2.5-2B` |
| `Ovis2.5-9B` |
| `Qwen2.5-VL-32B-Instruct` |
| `Qwen2.5-VL-3B-Instruct` |
| `Qwen2.5-VL-72B-Instruct` |
| `Qwen2.5-VL-7B-Instruct` |
| `Qwen3-VL-30B-A3B-Instruct` |
| `Qwen3-VL-32B-Instruct` |
| `Qwen3.5-27B` |
| `Qwen3.5-9B` |
| `Qwen3.6-27B` |
| `Qwen3.6-35B-A3B` |
| `GLM-4.1V-9B-Thinking` |
| `InternVL3-14B-Instruct` |
| `InternVL3_5-30B-A3B` |
| `Ovis2-16B` |
| `Ovis2-34B` |
| `Qwen3-VL-4B-Instruct` |
| `Qwen3-VL-8B-Instruct` |
| `InternVL3-78B` |
| `Ovis2.6-30B-A3B` |
| `Qwen3-VL-2B-Instruct` |
| `MiniCPM-V-4.6` |

</details>

The Figure 12 response table contains 158 model–query cells with multiple score records.

## 2. Figures 10–11: query-level estimation

### Historical evidence and selection

Historical run seeds are **0–4**; query-selection seeds are **0–2**. For each run, a seeded uniform permutation orders the 459,072 historical cells. The first 4,590 and 22,953 cells form nested 1% and 5% visibility masks. Internal development uses the declared rows of the same masked matrix.

Hidden scores remain **NaN**. Only observed historical scores enter fitting, Q construction, features, and encoder supervision. Evaluation models are excluded from historical fitting and development. Estimation for each evaluation model receives only its queried responses.

- **Figure 10 — shared IRT-selected queries:** allocate the same benchmark quotas, then cluster raw scalar IRT parameters $`(a_x,b_x)`$ within each benchmark. K-Means uses squared Euclidean distance, k-means++ initialization, 10 initializations, Lloyd iterations, `max_iter=300`, and `tol=1e-4`. Select unique actual queries nearest the centroids, breaking ties by stable query identifier. Each budget has its own query set.
- **Figure 11 — stratified sampling:** select from the benchmark with the smallest current count, using seeded ties and within-benchmark permutations. Budgets are prefixes of one sequence per evaluation model and query seed.

The learned methods receive the same observations for each evaluation model. Stratified sequences are reused across historical runs and visibility levels; IRT-selected query sets are shared across evaluation models within each run, visibility, query seed, and budget.

### Methods and training

| Estimator | Historical fitting | Estimation for an evaluation model |
|---|---|---|
| **Ours / NC2PL-FE** | Eight-dimensional conjunctive decoder and ability encoder | One encoder forward pass through the frozen decoder |
| **IRT** [[1]](#ref-1) | Scalar hierarchical 2PL | Scalar MAP with a standard-normal ability prior |
| **PSN-IRT** [[2]](#ref-2) | Scalar neural 2PL, hidden width 64 | Same scalar MAP solver |
| **Random** | None | Separate uniform query sample for each evaluation model; its global mean estimates every benchmark |
| **Historical mean** | Pool visible historical scores within each benchmark | No evaluation-model queries; use the global historical mean if a benchmark has no support |

For eight-dimensional ability $`\boldsymbol\theta_i`$, Ours predicts query scores as

```math
\rho_x(\boldsymbol\theta_i)
=\prod_{h:Q_{xh}=1}\sigma(a_{xh}\theta_{ih}-\beta_{xh}),
\qquad a_{xh}\geq10^{-6}.
```

Here $`\sigma`$ is the logistic function and binary $`Q`$ selects active factors.

**Q construction.** Fit an eight-dimensional additive MIRT model to the permitted historical scores. Retain the largest absolute loading, plus the second when it is at least half the largest. Ties use seeded factor permutations. A query without historical support receives one seeded factor independent of scores. Only the binary Q matrix transfers to the decoder and remains fixed.

**Decoder.** With fractional cross-entropy $`\ell(y,p)`$ and observed coordinates $`\mathcal O_q`$, minimize

```math
\mathcal L_{\mathrm{dec}}=
\frac{
\sum_{(i,x)\in\mathcal O_q}\ell(y_i(x),\rho_x(\boldsymbol\theta_i))
+\frac12\sum_i\|\boldsymbol\theta_i\|^2
+\frac12\sum_{x,h:Q_{xh}=1}\bigl[(a_{xh}-1)^2+\beta_{xh}^2\bigr]
}{|\mathcal O_q|}.
```

Initialize abilities at zero, active discriminations at one, and biases at zero. Project discriminations to the lower bound after updates; unsupported query parameters stay at their prior modes.

| Component | Optimizer and training settings |
|---|---|
| MIRT-to-Q | Adam; 2,000 updates; learning rate 0.03; sample $`\min(64M,\lvert\mathcal O_q\rvert)`$ observed coordinates with replacement for $`M`$ fitting models |
| Conjunctive decoder | Adam; 2,000 updates; learning rate 0.03; batch 4,096 observed coordinates with replacement, or all coordinates when the observed count is at most 4,096 |
| IRT | Hierarchical SVI; 2,000 full-observation updates; Adam learning rate 0.03, multiplied by 0.9999 per update; retain the lowest-ELBO-loss guide locations |
| PSN-IRT | AdamW; 100 full epochs; learning rate 0.0003; weight decay 0.00001; observed-coordinate batch 4,096 |

**Encoder features.** Let $`M`$ be the fixed reference-coordinate count (12 or 18) and $`n_x`$ the support for query $`x`$ after row exclusion and dropout. Excluded rows retain zeroed coordinates. The reference bank stores centered failure scores, availability, and five statistics: mean failure, population variance, normalized support $`\log(1+n_x)/\log(1+M)`$, zero-support flag, and support-below-two flag. Zero-support means and variances use flagged zero placeholders. Context features average these blocks with failure and success weights $`(1-y)`$ and $`y`$, divided by context size $`B`$. They also include the model's mean score on the context queries, $`\log(1+B)/\log(401)`$, and 18 alphabetically ordered benchmark fractions. The input dimension is $`4M+30`$: **78 during development and 102 for final fitting**. The network is two 64-unit GELU layers and an unconstrained eight-dimensional linear output.

**Encoder training.** Sample a historical model to serve as the model being characterized, remove its entire row from the reference bank, and drop each remaining reference row with probability 0.2. Sample a feasible context budget from **20, 40, 100, 200, 400**. Use disjoint observed query sets for the context and supervision (up to 512 queries for supervision). Train with mean fractional cross-entropy through the frozen decoder, one AdamW update per episode and 256 episodes per epoch.

Development compares learning rates **0.001 / 0.0003** and weight decays **0 / 0.0001**, using fixed episodes, at most 100 epochs, and patience 10. Select by pooled **MSE** on the supervision queries, then reinitialize and refit on all 18 historical models for the selected number of epochs.

Training cross-entropy, development MSE, and final DatasetScoreMAE are distinct quantities. All displayed budgets reuse frozen models, including **5, 10, and 15**, which are below the encoder’s training-context budgets.

### Evaluation

For learned estimators, retain each evaluation model's observed query scores and use model predictions for unqueried positions. Let $`\widehat s_{*,g}`$ be the completed mean for benchmark $`g`$ and $`s_{*,g}`$ its full-reference mean. Compute

```math
\mathrm{DatasetScoreMAE}
=\frac1{18}\sum_{g=1}^{18}|\widehat s_{*,g}-s_{*,g}|.
```

Queries have equal weight within benchmarks; benchmarks have equal weight in MAE. Random and Historical mean use the direct estimates in the method table.

Average the three query seeds within each evaluation model, then the seven evaluation models within each run, then the five runs. Reported run SD is the sample standard deviation of the five run means. Random is reused across historical conditions and both selection panels. Historical mean averages seven evaluation models and five runs with no query-seed replication; its line is constant across budgets.

## 3. Figure 12: cross-domain prediction

Behavioral clustering groups 25,457 queries into **32 fixed domains**, using responses from all 35 models. Evaluation models also participated in development, so domain construction is transductive. Each supplied historical score or anchor score for an evaluation model is a full-domain mean. Methods share one historical mask per missingness condition and seeded, nested anchor-domain sets for each evaluation model. Historical standardization statistics and predictor parameters remain fixed while predicting scores for the evaluation models.

| Predictor | Configuration |
|---|---|
| **Graph (ours)** | Score-aware bipartite graph; two width-16 message-passing layers; residual updates, GELU, dropout, and LayerNorm; bilinear decoder with biases; 100 Huber-loss training updates with observed edges masked before message passing; 0–4 noisy training anchor scores |
| **History mean / Mean + offset** | Historical domain means, with or without an offset from each evaluation model's anchor scores |
| **Raw Bias ALS / BenchPress adaptation** | Rank-two factorization in raw-score / standardized logit space; regularization 0.1; relative stopping tolerance $`10^{-6}`$ |
| **PMF-MAP / PMF-MCMC** | Rank-two probabilistic factorization; MAP factors / evaluation-model factors conditioned on frozen historical posterior draws; MCMC uses four chains, 3,000 burn-in and 2,000 retained draws per chain |

Graph predictions are inverse-standardized and clipped to [0,1]. For measured-domain set $`\mathcal J_{*}(B)`$, evaluate only unmeasured domains, using estimated scores $`\widehat s_{*j}`$ and full-reference scores $`s_{*j}`$:

```math
\mathrm{DomainMAE}
=\frac1{32-B}\sum_{j\notin\mathcal J_{*}(B)}|\widehat s_{*j}-s_{*j}|.
```

Each condition and budget averages 45 evaluations: three fitting seeds × five evaluation models × three selection seeds. Figure 12(a) then averages the four missingness conditions equally; Figure 12(b) fixes **B=4**. The test budget includes eight domains with exact anchor scores; Graph trains with zero to four noisy anchor scores. Some PMF-MCMC chains mix inadequately, so their means are descriptive. These anchor scores are complete-domain means and do not measure query-level estimation error.

## References

<a id="ref-1"></a>

[1] Polo et al. (2024). [tinyBenchmarks: evaluating LLMs with fewer examples](https://proceedings.mlr.press/v235/maia-polo24a.html).

<a id="ref-2"></a>

[2] Zhou et al. (2026). [Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](https://ojs.aaai.org/index.php/AAAI/article/view/40814).
