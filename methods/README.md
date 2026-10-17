# Public routing baselines

This directory contains the **12 public baselines ranked in report §6.3** and their shared
scoring interface. Mio selector is replay-only; its internal trainer is not included.

Implementations derive from [ORBIT](https://github.com/LAMDA-Model-Reuse/ORBIT), commit
`494810de2605f69737e72b55baf6e60c95c6dec0`, under its retained [MIT license](orbit/LICENSE).
The table describes the actual fitters used here. Embedding-only adaptations can differ
from the original paper's inputs, architecture, or objective; they are not claims of exact
paper reproduction. [The registry](configs/orbit_registry.yaml) records integration metadata.

## Implementations

| Method reference | ID | Implementation |
| --- | --- | --- |
| [Avengers](https://arxiv.org/abs/2505.19797) | `avengers` | `fit_avengers`: Cluster means of training quality, cost, and latency; nearest-cluster prediction. |
| [Avengers-Pro](https://arxiv.org/abs/2508.12631) | `avengerspro` | `fit_avengers`: The same clustering fitter as Avengers, with configurable multi-cluster averaging. |
| [Causal RM-Softmax](https://arxiv.org/abs/2505.16037) | `causal_rm_softmax` | `fit_rm_softmax`: Regret minimization over cost/latency trade-offs plus MSE calibration of three score heads. |
| [Eagle](https://arxiv.org/abs/2409.15518) | `eagle` | `fit_global_local`: A weighted blend of global mean labels and distance-weighted kNN predictions. |
| [EmbedLLM-RouteJudge](https://arxiv.org/abs/2410.02223) | `embedllm_routejudge` | `fit_embedllm`: Learned query/model interactions with BCE quality loss and MSE resource heads; resource predictions depend on model identity. |
| [GraphRouter](https://arxiv.org/abs/2410.03834) | `graphrouter` | `fit_graphrouter`: Query/model star-graph edge prediction with training-derived capability descriptors and three output scores. |
| [HybridLLM](https://arxiv.org/abs/2404.14618) | `hybridllm` | `fit_ridge`: Independent multi-output ridge regressors on the bundled query embeddings. |
| [kNN](https://arxiv.org/abs/2408.12320) | `knn` | `OrbitKNNEstimator`: Distance-weighted kNN for all three label matrices; exact matches share the weight. |
| [MLP](https://arxiv.org/abs/2408.12320) | `mlp` | `fit_mlp`: Independent MLP regressors for quality, cost, and latency; MSE loss. |
| [MIRT](https://aclanthology.org/2025.acl-long.761/) | `mirt` | `fit_mirt`: Training-derived model descriptors and query-conditioned item-response predictors. Sigmoid outputs bound all three score heads to [0, 1]. |
| [RouteLLM-MF](https://arxiv.org/abs/2406.18665) | `routellm_mf` | `fit_low_rank`: SVD of training-label matrices followed by ridge prediction of latent query factors; an MF-style approximation. |
| [SVM Router](https://arxiv.org/abs/2408.12320) | `svm_router` | `fit_svm`: Independent LinearSVR models for each candidate and each of the three labels. |

Avengers and Avengers-Pro share a clustering fitter with different averaging settings.
Historical aliases are excluded from the report's 12-baseline inventory.
Frozen configurations are in [run_report_baselines.py](../examples/run_report_baselines.py);
training dispatch and shared code are in [orbit/](orbit/training.py).

## Shared interface

`RoutingTargets` contains aligned training performance, cost, and latency matrices.
At inference, `ScoringRouter` receives query embeddings and returns `RoutingScores` for
all candidate models; `SelectionPolicy` then selects a model. All score matrices have
shape `[queries, models]`. Test outcomes never enter prediction.

```python
from benchmark.data import load_routing_split
from methods import train_orbit_router

train = load_routing_split("text", "train")
router = train_orbit_router(
    method="routellm_mf",
    query_embeddings=train.embeddings,
    performance_scores=train.performance,
    cost_scores=train.cost_cny,
    latency_scores=train.latency_ms,
    model_ids=train.model_ids,
)
```

See the [reproduction guide](../docs/reproducibility.md) for fitting all baselines and
using the official test split, nominal-budget metric, and scenario-specific units.
