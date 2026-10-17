"""Fit the report's 12 distinct public baselines from the public data interface.

This command trains on train only. The official test split is scored by the
report evaluator. Mio training is private and uses a separate prediction replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark.data import DEFAULT_REVISION, load_routing_split, materialize_routing_split
from evaluation.report_reproduction import evaluate_saved_predictions, source_group
from methods import train_orbit_router

METHODS = (
    "avengers",
    "avengerspro",
    "causal_rm_softmax",
    "eagle",
    "embedllm_routejudge",
    "graphrouter",
    "hybridllm",
    "knn",
    "mlp",
    "mirt",
    "routellm_mf",
    "svm_router",
)


def baseline_config(method: str, models: int, seed: int, device: str) -> dict:
    config = {
        "seed": seed,
        "device": device,
        "k": 32,
        "rank": min(8, models),
        "alpha": 0.05 if method == "embedllm_routejudge" else 1.0,
        "embed_dim": 64,
        "max_iter": 300,
        "router_hidden": [128, 64],
        "num_lambdas": 4,
        "model": {
            "hidden_sizes": [128, 64],
            "hidden_dim": 64,
            "num_layers": 1,
            "latent_dim": 16,
            "clusters": 64,
            "n_init": 1,
            "max_iter": 60,
            "algorithm": "lloyd",
            "dropout": 0.1,
            "activation": "relu",
        },
        "training": {
            "epochs": 2 if method == "mirt" else 5,
            "batch_size": 256,
            "lr": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "score_loss_weight": 1.0,
        },
        "selection": {"performance_weight": 1.0, "cost_weight": 0.0, "latency_weight": 0.0},
    }
    if method == "avengerspro":
        config["multi_cluster"] = 3
    if method == "eagle":
        config["local_weight"] = 0.65
    return config


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("text", "multimodal", "chinese"), required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = {}
    for split in ("train", "validation", "test"):
        if args.local_root:
            import pyarrow.parquet as pq

            path = args.local_root / args.scenario / f"{split}-00000-of-00001.parquet"
            data[split] = materialize_routing_split(
                pq.read_table(path).to_pylist(),
                scenario=args.scenario,
                split=split,
            )
        else:
            data[split] = load_routing_split(args.scenario, split, revision=args.revision)
    train, validation, test = data["train"], data["validation"], data["test"]
    if not (train.model_ids == validation.model_ids == test.model_ids):
        raise ValueError("Candidate-model order differs between splits")
    feature_provenance = None
    if args.scenario == "multimodal":
        from benchmark.data.feature_provenance import (
            validate_features,
            validate_multimodal_identity,
        )

        for split_data in data.values():
            for row in split_data.metadata:
                declared = json.loads(str(row.get("provenance_json") or "{}"))
                declared = declared.get("embedding")
                dimension = validate_multimodal_identity(declared)
                if row["embedding_model"] != declared["model_id"]:
                    raise ValueError("Stored features and declared encoder differ")
                if feature_provenance is not None and declared != feature_provenance:
                    raise ValueError("Methods and splits must share one frozen feature identity")
                feature_provenance = declared
            validate_features(split_data.embeddings, split_data.num_queries, dimension)
    train_groups = {
        source_group(query, row, args.scenario)
        for query, row in zip(train.query_ids, train.metadata, strict=True)
    }
    evaluation_groups = {
        source_group(query, row, args.scenario)
        for query, row in zip(test.query_ids, test.metadata, strict=True)
    }
    if train_groups & evaluation_groups:
        raise ValueError("Training and evaluation source groups overlap")
    validation_groups = {
        source_group(query, row, args.scenario)
        for query, row in zip(validation.query_ids, validation.metadata, strict=True)
    }
    if validation_groups & (train_groups | evaluation_groups):
        raise ValueError("Validation source groups must be separate from train and test")
    cost_unit = "CNY" if args.scenario == "chinese" else "USD"
    cost = np.asarray(train.cost_cny / (7.2 if cost_unit == "USD" else 1), dtype=np.float32)
    latency = np.asarray(train.latency_ms, dtype=np.float32)
    cost_scale = float(np.percentile(cost, 95))
    latency_scale = float(np.percentile(latency, 95))
    if not np.isfinite([cost_scale, latency_scale]).all() or min(cost_scale, latency_scale) <= 0:
        raise ValueError("Training-derived resource scales must be positive and finite")
    args.output.mkdir(parents=True)
    manifest = {
        "scenario": args.scenario,
        "seed": args.seed,
        "methods": args.methods,
        "data_revision": args.revision if not args.local_root else "local_staging",
        "train_feature_sha256": array_hash(train.embeddings),
        "train_performance_sha256": array_hash(train.performance),
        "cost_scale": cost_scale,
        "latency_scale": latency_scale,
        "cost_unit": cost_unit,
        "feature_dimension": train.embeddings.shape[1],
        "validation_used_for_baseline_fitting": False,
        "training_evaluation_source_group_overlap": 0,
        "python_version": platform.python_version(),
    }
    from importlib.metadata import version

    manifest["package_versions"] = {
        name: version(name) for name in ("numpy", "scikit-learn", "torch", "joblib", "pyarrow")
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for method in args.methods:
        config = baseline_config(method, train.num_models, args.seed, args.device)
        print(f"FIT {args.scenario} seed={args.seed} {method}", flush=True)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            router = train_orbit_router(
                method,
                train.embeddings,
                train.performance,
                cost / cost_scale,
                latency / latency_scale,
                train.model_ids,
                config,
            )
        fitted_seconds = perf_counter() - start
        start = perf_counter()
        predicted = router.predict_scores(test.embeddings)
        prediction_seconds = perf_counter() - start
        path = args.output / f"{method}.npz"
        np.savez_compressed(
            path,
            query_ids=np.asarray(test.query_ids),
            model_ids=np.asarray(router.model_ids),
            performance_scores=np.clip(predicted.performance_scores, 0, 1).astype(np.float32),
            cost_scores=np.asarray(predicted.cost_scores * cost_scale, dtype=np.float32),
            latency_scores=np.asarray(predicted.latency_scores * latency_scale, dtype=np.float32),
            policy=np.asarray([1.0, 0.0, 0.0, np.nan, np.nan]),
        )
        result = evaluate_saved_predictions(path, test, validation, cost_unit=cost_unit)
        result["experiment_identity"] = "fresh_baseline_fit_from_public_training_data"
        result["training_seconds"] = fitted_seconds
        result["prediction_seconds"] = prediction_seconds
        result["fitting_warnings"] = dict(
            Counter(f"{warning.category.__name__}: {warning.message}" for warning in fit_warnings)
        )
        result["predicted_quality_clipped_to_unit_interval"] = True
        result["feature_dimension"] = train.embeddings.shape[1]
        result["embedding_model"] = train.metadata[0]["embedding_model"]
        result["train_feature_sha256"] = manifest["train_feature_sha256"]
        result["train_performance_sha256"] = manifest["train_performance_sha256"]
        if feature_provenance is not None:
            result["feature_provenance"] = feature_provenance
        result["config"] = config
        result["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / f"{method}.json").write_text(json.dumps(result, indent=2) + "\n")
        router.save(args.output / f"{method}.joblib")
        print(
            json.dumps(
                {
                    "method": method,
                    "training_seconds": fitted_seconds,
                    "nauc": {
                        key: value["normalized_auc"] for key, value in result["splits"].items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
