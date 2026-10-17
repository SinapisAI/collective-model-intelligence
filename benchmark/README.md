# RobustRouteBench

The dataset is hosted at [SinapisAI/RobustRouteBench](https://huggingface.co/datasets/SinapisAI/RobustRouteBench).
The loader pins an immutable dataset revision for reproducible experiments.
It contains **35,072 scenario-level instances** across Text (63 models), Multimodal (45),
and Chinese (10). These are separate candidate pools, not a unique-model total.

```python
from benchmark.data import load_robust_route_bench, load_routing_split

raw = load_robust_route_bench("text", "validation")  # Hugging Face Dataset
train = load_routing_split("multimodal", "train")  # aligned NumPy matrices
print(train.embeddings.shape, train.performance.shape)
# (4000, 1024) (4000, 45)
```

After installing `.[data]`, `cmi-data text --split validation` prints the split size,
candidate count, encoder, and units. Supported splits are `train`, `validation`, and `test`.
Downloaded files belong in ignored `benchmark/data/downloads/` or the Hugging Face cache.

## Data schema and features

Each row contains `query_id`, `model_ids`, `embedding`, `performance`, `cost_cny`,
`latency_ms`, source/domain identifiers, evaluation/variation types, and provenance.
Candidate arrays follow the row's model order; normalized performance is in [0, 1].
Costs are **CNY/request** and latency is **milliseconds**. Source failures retain zero
performance. Raw model responses and failure diagnostics are not published.

Text and Chinese use 1,024-dimensional BGE-M3 features. Multimodal uses CLIP ViT-B/32
(`openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`):
pool normalized full-text chunks of up to 75 payload tokens by token count, pool all
normalized image vectors equally, normalize each modality pool, concatenate, and divide
by √2. The result is a unit-norm 1,024-vector shared by every method and fitting seed.
The loader checks the encoder identity, fusion protocol, feature-archive checksum identity,
and vector norms. Training uses these bundled features and needs no encoder weights or images.

Original images and accompanying metadata are optional for fitting/replay with bundled features.
To inspect images, rebuild features, or re-evaluate visual models, use
`download_multimodal_media` from `benchmark.data`:

| Subset argument | Archive | Role |
| --- | --- | --- |
| `"train"` | `media/multimodal/train.zip` | Training |
| `"validation"` | `media/multimodal/validation.zip` | Configuration selection |
| `"standard"` | `media/multimodal/standard_test.zip` | Standard test |
| `"robust"` | `media/multimodal/robust_test.zip` | Robustness test |

For example, `download_multimodal_media("standard")` downloads the standard-test archive
from the same pinned dataset repository. Filenames and checksums are recorded in the
dataset manifest. Each archive contains its role's requests, images, and reference annotations.
Preserve the dataset's [upstream licenses](https://huggingface.co/datasets/SinapisAI/RobustRouteBench/blob/03bd3da2c3bb4a998f73d285c8853c721697e3cb/LICENSES.md).

## Evaluation split

Every scenario provides `train`, `validation`, and `test`. Validation contains 500 requests
(400 standard + 100 robust) for configuration selection. All **18,982 test requests** are
scored directly, with no additional filtering. The test split is separate from validation
by request ID and the declared source-group identifiers.

| Scenario | Train | Validation | Standard test | Robust test |
| --- | ---: | ---: | ---: | ---: |
| Text | 3,000 | 500 | 3,688 | 2,894 |
| Multimodal | 4,000 | 500 | 4,600 | 1,900 |
| Chinese | 7,590 | 500 | 3,900 | 2,000 |

See [split discipline and units](../docs/reproducibility.md#split-discipline).
