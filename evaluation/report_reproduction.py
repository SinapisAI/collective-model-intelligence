"""Re-evaluate saved, ID-bearing predictions using the report's metric.

This reproduces evaluation, not private selector training. No model weights,
response execution, or live model API are needed for prediction replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark.data.huggingface import (
    DEFAULT_REVISION,
    RoutingSplit,
    load_robust_route_bench,
    materialize_routing_split,
)
from evaluation.metrics import EvaluationOutcomes, ReportBudgetCurveEvaluator
from methods.base import RoutingScores, SelectionPolicy


def source_group(query_id: str, row: dict, scenario: str) -> str:
    value = (
        row.get("source_sample_id")
        if scenario == "multimodal"
        else (row.get("base_query_id") or query_id)
    )
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing source group: {query_id}")
    return value


def load_predictions(path: Path, query_ids: tuple[str, ...]):
    """Never infer prediction row order from its length or evaluation scores."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "query_ids",
            "model_ids",
            "performance_scores",
            "cost_scores",
            "latency_scores",
            "policy",
        }
        if missing := required - set(data.files):
            raise ValueError(f"Prediction archive lacks identity-bearing arrays: {sorted(missing)}")
        ids = tuple(str(value) for value in data["query_ids"].tolist())
        if len(ids) != len(set(ids)) or set(ids) != set(query_ids):
            raise ValueError("Prediction query IDs must match the test split exactly")
        index = {query_id: position for position, query_id in enumerate(ids)}
        order = np.asarray([index[query_id] for query_id in query_ids])
        models = tuple(str(value) for value in data["model_ids"].tolist())
        if not models or len(set(models)) != len(models):
            raise ValueError("Prediction candidate IDs must be nonempty and unique")
        matrices = []
        for key in ("performance_scores", "cost_scores", "latency_scores"):
            matrix = np.asarray(data[key], dtype=np.float32)
            if matrix.shape != (len(ids), len(models)):
                raise ValueError(f"Prediction matrix shape mismatch: {key}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite prediction values: {key}")
            if key != "performance_scores" and (matrix < 0).any():
                raise ValueError(f"Negative resource predictions: {key}")
            matrices.append(matrix[order])
        values = np.asarray(data["policy"], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values[:3]).all():
            raise ValueError("Prediction policy must contain three finite weights and two limits")
        if np.any(values[:3] < 0):
            raise ValueError("Prediction utility weights must be nonnegative")
        limits = [None if np.isnan(value) else float(value) for value in values[3:]]
        if any(value is not None and not np.isfinite(value) for value in limits):
            raise ValueError("Prediction policy limits must be finite or NaN for no limit")
        policy = SelectionPolicy(*values[:3].tolist(), *limits)
    return RoutingScores(models, *matrices), policy


def evaluate_saved_predictions(
    predictions: Path,
    test: RoutingSplit,
    validation: RoutingSplit,
    *,
    cost_unit: str,
) -> dict:
    """Evaluate all official test requests without filtering either split."""
    if test.scenario != validation.scenario or test.model_ids != validation.model_ids:
        raise ValueError("Test and validation must share scenario and candidate-model order")
    if test.split != "test" or validation.split != "validation":
        raise ValueError("Expected test and validation splits")
    expected_currency = "CNY" if test.scenario == "chinese" else "USD"
    if cost_unit != expected_currency:
        raise ValueError(f"Report {test.scenario} prediction costs must use {expected_currency}")
    scores, policy = load_predictions(predictions, test.query_ids)
    if set(scores.model_ids) != set(test.model_ids):
        raise ValueError("Prediction candidate IDs do not match Hugging Face data")
    scenario = test.scenario
    validation_groups = {
        source_group(query_id, row, scenario)
        for query_id, row in zip(validation.query_ids, validation.metadata, strict=True)
    }
    test_groups = {
        source_group(query_id, row, scenario)
        for query_id, row in zip(test.query_ids, test.metadata, strict=True)
    }
    if set(test.query_ids) & set(validation.query_ids):
        raise ValueError("Test and validation query IDs must be disjoint")
    if test_groups & validation_groups:
        raise ValueError("Test and validation source groups must be disjoint")
    for row in test.metadata:
        if row.get("evaluation_type") not in {"standard", "robust"}:
            raise ValueError("Every test request must be standard or robust")
        if row["evaluation_type"] == "robust" and (
            not isinstance(row.get("variant_type"), str)
            or not row["variant_type"].strip()
            or row["variant_type"] == "original"
        ):
            raise ValueError("Every robust request must declare a non-original variant_type")
    slices = {
        name: np.asarray(
            [index for index, row in enumerate(test.metadata) if row["evaluation_type"] == name],
            dtype=np.int64,
        )
        for name in ("standard", "robust")
    }
    variants = sorted({test.metadata[int(index)]["variant_type"] for index in slices["robust"]})
    for variant in variants:
        slices[f"robust_{variant}"] = np.asarray(
            [
                index
                for index in slices["robust"]
                if test.metadata[int(index)]["variant_type"] == variant
            ],
            dtype=np.int64,
        )
    # Convert evaluator-side observed costs back to the report's original unit.
    # Saved prediction utilities and feasibility thresholds stay in that unit.
    actual_cost = test.cost_cny / (7.2 if cost_unit == "USD" else 1.0)
    evaluator = ReportBudgetCurveEvaluator(100)
    results = {}
    for name, indices in slices.items():
        if not indices.size:
            raise ValueError(f"Empty report population: {name}")
        subset = RoutingScores(
            scores.model_ids,
            scores.performance_scores[indices],
            scores.cost_scores[indices],
            scores.latency_scores[indices],
        )
        actual = EvaluationOutcomes(
            test.model_ids,
            test.performance[indices],
            actual_cost[indices],
            test.latency_ms[indices],
        )
        result = evaluator.evaluate_scores(subset, policy, actual)
        results[name] = {"query_count": len(indices), **result.to_dict()}
    return {
        "metric_identity": "report_nominal_budget_cumulative_maximum",
        "experiment_identity": "saved_prediction_re_evaluation_not_refitting",
        "scenario": scenario,
        "cost_unit": cost_unit,
        "splits": results,
        "split_audit": {
            "test_rows": test.num_queries,
            "validation_rows": validation.num_queries,
            "validation_test_query_overlap": 0,
            "validation_test_source_group_overlap": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("text", "multimodal", "chinese"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unverified-candidate", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.data_root is not None:
        import pyarrow.parquet as pq

        audit_path = args.data_root / "release-audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if not audit.get("release_integrity_verified") and not args.allow_unverified_candidate:
            raise ValueError("Missing release integrity audit; diagnostic opt-in is required")
        rows = {
            split: pq.read_table(args.data_root / f"{split}-00000-of-00001.parquet").to_pylist()
            for split in ("test", "validation")
        }
    else:
        from huggingface_hub import hf_hub_download

        from benchmark.data.huggingface import DATASET_ID

        try:
            audit_path = hf_hub_download(
                DATASET_ID, "release-audit.json", repo_type="dataset", revision=args.revision
            )
            audit = json.loads(Path(audit_path).read_text())
        except Exception:
            if not args.allow_unverified_candidate:
                raise ValueError(
                    "The pinned data revision lacks a release integrity audit"
                ) from None
            audit = {}
        if not audit.get("release_integrity_verified") and not args.allow_unverified_candidate:
            raise ValueError("Release integrity validation has not completed")
        rows = {
            split: load_robust_route_bench(args.scenario, split, revision=args.revision)
            for split in ("test", "validation")
        }
    data = {
        split: materialize_routing_split(value, scenario=args.scenario, split=split)
        for split, value in rows.items()
    }
    result = evaluate_saved_predictions(
        args.predictions,
        data["test"],
        data["validation"],
        cost_unit="CNY" if args.scenario == "chinese" else "USD",
    )
    result["release_integrity_verified"] = bool(audit.get("release_integrity_verified"))
    # Integrity and exact-match restoration do not independently re-grade every answer.
    result["all_outcomes_independently_regraded"] = False
    result["data_revision"] = args.revision if args.data_root is None else "local_candidate"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split_audit": result["split_audit"],
                "nauc": {name: row["normalized_auc"] for name, row in result["splits"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
