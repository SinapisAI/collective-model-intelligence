# Experimental results

[← Project home](../README.md) · [Reproduction guide](reproducibility.md)

Results for Sections 6.1–6.4 of the [technical report](../tech-report.pdf).

## Mio 1.0 Lite: public benchmarks

Report §6.1 compares **Mio 1.0 Lite**, eight fixed models, and four routers on nine public benchmarks. Mio combines task-dependent model selection and collaboration among cost-efficient models from China. The public subsets of AutomationBench and AA-OmniScience are used; BFCL is V3 Base Multi-Turn, GPQA is Diamond, and HLE is without tools.

| Benchmark | Mio score | Mean CNY/request |
| --- | ---: | ---: |
| AutomationBench (public) | 53.83% | 0.0792 |
| PinchBench | 96.27% | 0.0836 |
| QwenClawBench | 69.38% | 0.1651 |
| BFCL V3 Base Multi-Turn | 77.50% | 0.0186 |
| MT-Bench | 9.138 / 10 | 0.0245 |
| IFBench | 92.00% | 0.0219 |
| GPQA Diamond | 95.96% | 0.0239 |
| Humanity’s Last Exam (w/o tools) | 53.38% | 0.1273 |
| AA-OmniScience (public) | 64.67% | 0.0145 |

Mio’s dataset-equal mean is **77.15/100** and **0.0621 CNY/request**. Both are best among the 13 compared methods. It leads in score on AutomationBench, PinchBench, BFCL V3 Base, IFBench, and GPQA Diamond. Compared with the highest-scoring average baseline, GPT 6.1 Sol, it gains 6.35 points at 94.0% lower average cost. QwenClawBench and AA-OmniScience remain within 0.43 and 0.67 percentage points of the leaders, at 95.1% and 85.3% lower costs respectively.

![Mio 1.0 Lite quality–cost trade-offs on six public benchmarks, from report Figure 9.](assets/mio-public-performance.png)

Each benchmark contributes weight 1/9. Scores are normalized to 0–100 (MT-Bench × 10). Costs are normalized per request using each method's evaluation count, then averaged across the nine benchmarks. Mio's request cost sums the fees of the models used in collaboration. PinchBench and QwenClawBench use method-specific counts for some routers.

The report's Tables 2–4 list all **117 score–cost pairs** and 13 method averages. The source conversion convention is USD 1 = CNY 7. Scores and costs are rounded independently for display; MT-Bench 9.1375 displays as 9.138.

Mio 1.0 Lite's public benchmark results and the RobustRouteBench results below are separate studies. RobustRouteBench evaluates a request-level Mio selector trained separately for each scenario; its reproduction artifacts are described in the [reproduction guide](reproducibility.md).

## Model Eval: characterizing models with less evaluation

Report §6.2 studies limited-query score estimation and cross-domain prediction separately. See [Model Eval experimental settings](model-eval.md) for data, model inventories, training, and metrics.

### Limited-query score estimation

The fixed inventory has 25 models and 25,504 queries across 18 benchmarks. Eighteen models provide historical fitting data; seven are held out. Five runs expose 1% or 5% of historical scores. Each evaluation model has three query seeds, and learned methods share the same 5–40 observations from stratified sampling or IRT-selected query sets. Historical response functions remain frozen.

![Benchmark-score MAE versus observed query count under stratified sampling.](assets/model-eval.png)

Our method has the lowest mean MAE at every displayed budget under both selection policies and visibility levels. With stratified sampling, its mean MAE at five queries is lower than PSN-IRT’s at 40 queries under both visibility levels.

### Cross-domain prediction

Seven predictors receive shared exact anchor scores. Graph has the lowest average unmeasured-domain MAE at each nonzero displayed budget. At eight measured domains, MAE is **0.1474**, compared with **0.1597** for History mean, equally averaged over four historical missingness conditions.

![Cross-domain prediction MAE by measured-domain budget and historical missingness.](assets/domain-prediction.png)

Query populations are defined transductively using all 35 models, and the five evaluation models were seen during development. Graph trains with zero to four noisy anchor scores; eight domains with exact anchor scores extend beyond that range. Some PMF-MCMC chains mix inadequately. The two Model Eval studies are evaluated separately.

## RobustRouteBench: robustness is scenario dependent

Report §6.3 evaluates the request-level **Mio selector**, trained separately per scenario, against **12 baselines**. See the [split protocol](reproducibility.md#split-discipline) for the dataset identifier and official training, validation, and test populations.

All Multimodal methods share frozen **CLIP ViT-B/32** features: pool full-text chunks and all request images separately, normalize the pools, concatenate, and apply equal modality weights. This produces 1,024-dimensional late-fusion features. Multimodal uses three fits; General Text and Chinese use one.

| Scenario / split | Mio peak quality | Mio nAUC | nAUC rank / 13 |
| --- | ---: | ---: | ---: |
| General Text / standard | 0.5979 | 0.5924 | 4 |
| General Text / robust | 0.5596 | 0.5328 | 1 |
| Multimodal / standard | 0.7084 ± 0.0008 | 0.7077 ± 0.0007 | 1 |
| Multimodal / robust | 0.6266 ± 0.0008 | 0.6128 ± 0.0005 | 1 |
| Chinese / standard | 0.7711 | 0.7653 | 4 |
| Chinese / robust | 0.6137 | 0.6051 | 5 |

Multimodal metrics are computed **after pointwise averaging on the shared nominal-budget grid**. Their ± values are exact run-bootstrap standard errors, enumerating all 27 ordered resamples of the three curves. They are not sample standard deviations of per-run peaks or nAUCs. Macro nAUC is the equal mean of six split values: **0.6360 for Mio**, **0.6183 for MLP**.

![Supplementary RobustRouteBench quality–cost figure from the corrected report snapshot.](assets/robustroutebench.png)

The figure is retained as a supplementary artifact; the current report presents [overall](../tech-report.pdf) and [variation-level](../tech-report.pdf) tables. Mio leads Text Robust and both Multimodal splits, but not Text Standard or either Chinese split. Multimodal standard and robust populations are independently sampled, so their difference is not a paired perturbation effect.

nAUC integrates the cumulative-maximum quality curve on the **linear nominal-budget axis**, with no inserted fixed-model points. The supplementary figure displays realized cost on a logarithmic axis. These axes have different meanings; the badge is not the geometric area under that display. `Cost@τ` is the minimum realized cost of actual selector points meeting the target. [Metric definitions and runnable reproduction →](reproducibility.md#metric-compatibility)

Every official test request contributes to scoring; predictions are aligned by query ID. Baseline configurations are fixed while Mio configurations are selected using validation, with unmatched search budgets.

Data: [published curves, metrics, and split audit](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/03bd3da2c3bb4a998f73d285c8853c721697e3cb/report/aggregate-results.json). [Reproduction protocol](reproducibility.md).

## Mos: model-file distribution

Report §6.4 compares direct Alibaba Cloud OSS reads with a **prewarmed Fluid cache** on a self-managed six-node Kubernetes cluster. Each node has 64 vCPUs and 256 GB memory, with a 10 Gbps intranet. Loading pods have two vCPUs. Observed OSS download speeds are 33.1–53.5 MB/s. The cache uses Fluid 1.1.0-alpha and Curvine 0.5.2-alpha with one master and one worker.

Three models (0.62, 1.52, and 18.84 GB) are tested at concurrency 1, 3, 5, 7, and 10, with three repeats per condition. Durations are averaged across repeats before computing ratios.

![Batch model-file acquisition time by model and concurrency.](assets/model-distribution.png)

![Mean per-instance model-file read duration.](assets/model-distribution-per-instance.png)

| Model | Mean batch speedup | Mean per-instance speedup | Peak batch | Peak per-instance |
| --- | ---: | ---: | ---: | ---: |
| hayai-ocr-v2 | 8.73× | 9.04× | 20.67× | 19.90× |
| s1-mini | 9.39× | 9.68× | 16.70× | 16.36× |
| Ornith-1.0-9B | 12.00× | 10.84× | 22.56× | 23.31× |

Mean and peak summarize the five concurrency levels. Batch completion is measured from the common release signal until all validated pod results are collected; per-instance time averages pod-read durations. The cache is prewarmed, and timing ends at file retrieval into host memory, before GPU transfer and inference.

Full settings and the speedup summary appear in report §6.4. The cluster runner is not released.
