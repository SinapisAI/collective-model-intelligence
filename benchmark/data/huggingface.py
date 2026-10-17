"""Load the public, self-contained RobustRouteBench release from Hugging Face."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DATASET_ID = "SinapisAI/RobustRouteBench"
DEFAULT_REVISION = "03bd3da2c3bb4a998f73d285c8853c721697e3cb"
SCENARIOS = ("text", "multimodal", "chinese")
SPLITS = ("train", "validation", "test")
MULTIMODAL_MEDIA = {
    "train": "media/multimodal/train.zip",
    "validation": "media/multimodal/validation.zip",
    "standard": "media/multimodal/standard_test.zip",
    "robust": "media/multimodal/robust_test.zip",
}


@dataclass(frozen=True)
class RoutingSplit:
    """Materialized query inputs and aligned candidate-model labels."""

    scenario: str
    split: str
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    embeddings: NDArray[np.float32]
    performance: NDArray[np.float32]
    cost_cny: NDArray[np.float64]
    latency_ms: NDArray[np.float64]
    metadata: tuple[dict[str, object], ...]

    @property
    def num_queries(self) -> int:
        return self.embeddings.shape[0]

    @property
    def num_models(self) -> int:
        return self.performance.shape[1]


def _validate_scenario(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")


def _validate_split(split: str | None) -> None:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")


def load_robust_route_bench(
    scenario: str,
    split: str | None = None,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
    token: str | bool | None = None,
) -> Any:
    """Load one official scenario, optionally restricted to one published split.

    The pinned default revision makes experiments reproducible. Pass
    ``revision="main"`` explicitly to follow future dataset updates.
    """

    _validate_scenario(scenario)
    _validate_split(split)
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise RuntimeError("Hugging Face loading requires `pip install -e '.[data]'`") from error

    hub_kwargs: dict[str, object] = {
        "repo_id": DATASET_ID,
        "repo_type": "dataset",
        "revision": revision,
    }
    if cache_dir is not None:
        hub_kwargs["cache_dir"] = str(cache_dir)
    if token is not None:
        hub_kwargs["token"] = token

    requested_splits = (split,) if split is not None else SPLITS
    data_files = {}
    for requested_split in requested_splits:
        filename = f"data/{scenario}/{requested_split}-00000-of-00001.parquet"
        data_files[requested_split] = hf_hub_download(filename=filename, **hub_kwargs)

    dataset_kwargs: dict[str, object] = {
        "data_files": data_files,
        "streaming": streaming,
    }
    if cache_dir is not None:
        dataset_kwargs["cache_dir"] = str(cache_dir)
    if split is not None:
        dataset_kwargs["split"] = split
    return load_dataset("parquet", **dataset_kwargs)


def materialize_routing_split(
    rows: Sequence[Mapping[str, object]] | Any,
    *,
    scenario: str | None = None,
    split: str | None = None,
    embeddings: object | None = None,
    embedding_metadata: Mapping[str, object] | None = None,
) -> RoutingSplit:
    """Convert a loaded split into matrices accepted by the routing methods."""

    if len(rows) == 0:
        raise ValueError("cannot materialize an empty dataset split")
    if embeddings is not None and not (embedding_metadata or {}).get("embedding_model"):
        raise ValueError("replacement embeddings require their actual embedding_model metadata")
    if embeddings is None and embedding_metadata is not None:
        raise ValueError("embedding metadata cannot relabel unchanged bundled features")
    records = [dict(row) for row in rows]
    first = records[0]
    resolved_scenario = scenario or str(first["scenario"])
    resolved_split = split or str(first["published_split"])
    _validate_scenario(resolved_scenario)
    _validate_split(resolved_split)

    model_ids = tuple(first["model_ids"])
    if (
        not model_ids
        or any(not isinstance(value, str) or not value.strip() for value in model_ids)
        or len(set(model_ids)) != len(model_ids)
    ):
        raise ValueError("model_ids must be non-empty, unique strings")

    query_ids: list[str] = []
    stored_embeddings: list[object] = []
    performance: list[object] = []
    cost: list[object] = []
    latency: list[object] = []
    metadata: list[dict[str, object]] = []
    metadata_fields = (
        "source_split",
        "evaluation_type",
        "base_query_id",
        "is_validation",
        "domain",
        "source_dataset",
        "source_sample_id",
        "variant_type",
        "query_text",
        "input_json",
        "media_json",
        "provenance_json",
        "embedding_model",
        "embedding_revision",
        "embedding_instruction",
        "embedding_protocol",
        "embedding_dimension",
    )
    for index, row in enumerate(records):
        row_models = tuple(row["model_ids"])
        if row_models != model_ids:
            raise ValueError(f"row {index} model_ids do not match the split model order")
        row_scenario = str(row["scenario"])
        row_split = str(row["published_split"])
        if row_scenario != resolved_scenario or row_split != resolved_split:
            raise ValueError(f"row {index} belongs to {row_scenario}/{row_split}")
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"row {index} query_id must be a non-empty string")
        query_ids.append(query_id)
        if embeddings is None:
            stored_embeddings.append(row["embedding"])
        performance.append(row["performance"])
        cost.append(row["cost_cny"])
        latency.append(row["latency_ms"])
        record_metadata = {field: row.get(field) for field in metadata_fields}
        if embedding_metadata is not None:
            record_metadata["bundled_embedding_model"] = row.get("embedding_model")
            record_metadata.update(embedding_metadata)
        metadata.append(record_metadata)

    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique within a published split")
    matrices = {
        "embeddings": np.asarray(
            stored_embeddings if embeddings is None else embeddings, dtype=np.float32
        ),
        "performance": np.asarray(performance, dtype=np.float32),
        "cost_cny": np.asarray(cost, dtype=np.float64),
        "latency_ms": np.asarray(latency, dtype=np.float64),
    }
    expected_rows = len(records)
    expected_models = len(model_ids)
    features = matrices["embeddings"]
    if features.ndim != 2 or features.shape[0] != expected_rows or features.shape[1] < 1:
        raise ValueError("embeddings must have shape [num_queries, positive_feature_dimension]")
    if not np.isfinite(features).all():
        raise ValueError("embeddings must contain only finite values")
    for name in ("performance", "cost_cny", "latency_ms"):
        if matrices[name].shape != (expected_rows, expected_models):
            raise ValueError(f"{name} must have shape [{expected_rows}, {expected_models}]")
        if not np.isfinite(matrices[name]).all():
            raise ValueError(f"{name} must contain only finite values")
    if (matrices["cost_cny"] < 0).any() or (matrices["latency_ms"] < 0).any():
        raise ValueError("cost_cny and latency_ms must be non-negative")
    if (matrices["performance"] < 0).any() or (matrices["performance"] > 1).any():
        raise ValueError("normalized performance labels must lie in [0, 1]")

    return RoutingSplit(
        scenario=resolved_scenario,
        split=resolved_split,
        query_ids=tuple(query_ids),
        model_ids=model_ids,
        embeddings=matrices["embeddings"],
        performance=matrices["performance"],
        cost_cny=matrices["cost_cny"],
        latency_ms=matrices["latency_ms"],
        metadata=tuple(metadata),
    )


def load_routing_split(
    scenario: str,
    split: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> RoutingSplit:
    """Load report matrices, verifying the frozen CLIP features for Multimodal."""

    dataset = load_robust_route_bench(
        scenario, split, revision=revision, cache_dir=cache_dir, token=token
    )
    if scenario != "multimodal":
        return materialize_routing_split(dataset, scenario=scenario, split=split)

    from benchmark.data.feature_provenance import (
        CLIP_MODEL_ID,
        validate_features,
        validate_multimodal_identity,
    )

    records, identity = [], None
    for row in dataset:
        row = dict(row)
        if row.get("embedding_model") != CLIP_MODEL_ID:
            raise ValueError("The report requires the frozen CLIP multimodal features")
        declared = json.loads(str(row.get("provenance_json") or "{}")).get("embedding")
        dimension = validate_multimodal_identity(declared)
        if identity is not None and declared != identity:
            raise ValueError("Inconsistent bundled CLIP provenance")
        identity = declared
        row.update(
            embedding_revision=declared["revision"],
            embedding_protocol=declared["protocol"],
            embedding_dimension=dimension,
        )
        records.append(row)
    result = materialize_routing_split(records, scenario=scenario, split=split)
    validate_features(result.embeddings, result.num_queries, dimension)
    return result


def download_multimodal_media(
    subset: str,
    *,
    revision: str = DEFAULT_REVISION,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> Path:
    """Download optional media for train, validation, standard, or robust requests."""

    if subset not in MULTIMODAL_MEDIA:
        raise ValueError(
            f"multimodal subset must be one of {tuple(MULTIMODAL_MEDIA)}, got {subset!r}"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise RuntimeError("Media downloads require `pip install -e '.[data]'`") from error

    kwargs: dict[str, object] = {
        "repo_id": DATASET_ID,
        "repo_type": "dataset",
        "revision": revision,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if token is not None:
        kwargs["token"] = token
    manifest_path = Path(hf_hub_download(filename="manifest.json", **kwargs))
    manifest = json.loads(manifest_path.read_text())
    filename = manifest["media_archives"][subset]["path"]
    if (
        not isinstance(filename, str)
        or not filename.startswith("media/multimodal/")
        or not filename.endswith(".zip")
        or ".." in Path(filename).parts
    ):
        raise ValueError("Invalid multimodal archive path in the release manifest")
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    return Path(hf_hub_download(filename=filename, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(prog="cmi-data")
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    split = load_routing_split(
        args.scenario,
        args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "revision": args.revision,
                "scenario": split.scenario,
                "split": split.split,
                "queries": split.num_queries,
                "models": split.num_models,
                "embedding_dim": split.embeddings.shape[1],
                "embedding_model": split.metadata[0]["embedding_model"],
                "cost_currency": "CNY",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
