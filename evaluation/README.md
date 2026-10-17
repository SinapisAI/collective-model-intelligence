# Report evaluation

The evaluator implements report §6.3 under the same candidate pools, official test requests,
and score/cost conventions. Prediction arrays carry query and model IDs; their order is
checked before matching them to held-out outcomes.

- `metrics/report.py`: nominal-budget nAUC and target costs, built on the shared budget evaluator.
- `report_reproduction.py`: prediction alignment, split validation, units, and scenario/variation metrics.
- `report_aggregation.py`: pointwise averaging of fitted-run curves and exact run-bootstrap standard errors.
- `report_release.py`: complete report assembly, macro averages, ranks, and numerical comparisons.

```sh
python -m examples.reproduce_report --mode replay --output runs/report-replay
```

nAUC integrates cumulative-maximum quality over nominal budget, without inserting fixed-model
points. `Cost@τ` considers actual router points only. For Multimodal, average the three curves
before computing metrics. See the [protocol](../docs/reproducibility.md#metric-compatibility)
and [published results](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/03bd3da2c3bb4a998f73d285c8853c721697e3cb/report/aggregate-results.json).

Test outcomes enter scoring only. Budget feasibility uses predicted costs; the cheapest
predicted candidate is the fallback when no candidate is feasible. This does not guarantee
that realized cost or latency meets a nominal limit.
