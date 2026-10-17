"""Train kNN and inspect offline quality/cost metrics, without model API calls."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from benchmark.data import DEFAULT_REVISION, load_routing_split
from evaluation.metrics import EvaluationOutcomes, ReportBudgetCurveEvaluator
from methods import train_orbit_router


@dataclass
class ExampleSplit:
    embeddings: np.ndarray
    performance: np.ndarray
    cost_cny: np.ndarray
    latency_ms: np.ndarray
    model_ids: tuple[str, ...] = ("economy", "reasoning", "generalist")


def synthetic_split(rng: np.random.Generator, rows: int) -> ExampleSplit:
    embeddings = rng.normal(size=(rows, 8)).astype(np.float32)
    difficulty = 1.0 / (1.0 + np.exp(-embeddings[:, 0]))
    performance = np.column_stack(
        (0.9 - 0.6 * difficulty, 0.5 + 0.45 * difficulty, 0.7 + 0.1 * difficulty)
    )
    return ExampleSplit(
        embeddings=embeddings,
        performance=performance,
        cost_cny=np.tile([0.01, 0.08, 0.04], (rows, 1)),
        latency_ms=np.tile([100.0, 500.0, 250.0], (rows, 1)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("text", "multimodal", "chinese"))
    parser.add_argument("--cache-dir", default="benchmark/data/downloads")
    args = parser.parse_args()
    if args.scenario:
        load_kwargs = {"cache_dir": args.cache_dir}
        train = load_routing_split(args.scenario, "train", **load_kwargs)
        evaluation = load_routing_split(args.scenario, "validation", **load_kwargs)
        population = "validation (configuration selection; not test scoring)"
    else:
        rng = np.random.default_rng(42)
        train = synthetic_split(rng, 128)
        evaluation = synthetic_split(rng, 32)
        population = "synthetic demonstration; not measured benchmark evidence"

    config = {"k": 20, "seed": 42, "device": "cpu"}
    router = train_orbit_router(
        method="knn",
        query_embeddings=train.embeddings,
        performance_scores=train.performance,
        cost_scores=train.cost_cny,
        latency_scores=train.latency_ms,
        model_ids=train.model_ids,
        config=config,
    )
    # No evaluation outcomes enter the router: inference receives embeddings only.
    output = router.route_embeddings(evaluation.embeddings)
    outcomes = EvaluationOutcomes(
        model_ids=evaluation.model_ids,
        performance=evaluation.performance,
        cost=evaluation.cost_cny,
        latency=evaluation.latency_ms,
    )
    result = ReportBudgetCurveEvaluator().evaluate_scores(
        output.scores, router.selection_policy, outcomes
    )
    metrics = result.to_dict()
    summary = {
        "scenario": args.scenario or "synthetic",
        "population": population,
        "dataset_revision": DEFAULT_REVISION if args.scenario else None,
        "method": "knn",
        "config": config,
        "training_queries": len(train.embeddings),
        "evaluation_queries": len(evaluation.embeddings),
        "candidate_models": len(train.model_ids),
        "embedding_dimension": train.embeddings.shape[1],
        "embedding_model": train.metadata[0]["embedding_model"] if args.scenario else "synthetic",
        "cost_unit": "CNY per request",
        "latency_unit": "milliseconds (stored outcomes; not live service latency)",
        "metric_protocol": "report nominal-budget nAUC",
        "first_five_unconstrained_selections": output.selected_model_ids[:5],
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
