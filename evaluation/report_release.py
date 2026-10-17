"""Build report tables' single source of truth from complete per-run evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite

import numpy as np

from evaluation.report_aggregation import aggregate_report_runs

METHOD_NAMES = {
    "mio": "Mio selector (offline)",
    "avengers": "Avengers",
    "avengerspro": "Avengers-Pro",
    "causal_rm_softmax": "Causal RM-Softmax",
    "eagle": "Eagle",
    "embedllm_routejudge": "EmbedLLM-RouteJudge",
    "graphrouter": "GraphRouter",
    "hybridllm": "HybridLLM",
    "knn": "kNN",
    "mlp": "MLP",
    "mirt": "MIRT",
    "routellm_mf": "RouteLLM-MF",
    "svm_router": "SVM Router",
}
SCENARIO_SEEDS = {
    "text": (20260917,),
    "multimodal": (20260917, 20260918, 20260919),
    "chinese": (20260917,),
}
MODEL_COUNTS = {"text": 63, "multimodal": 45, "chinese": 10}
FINAL_COUNTS = {"text": (3688, 2894), "multimodal": (4600, 1900), "chinese": (3900, 2000)}


def assert_report_matches(actual: object, expected: object, location: str = "") -> None:
    """Compare frozen metrics/curves, rejecting nonfinite or mistyped numbers.

    Generation time, release-description metadata, and the separately checked
    split audit are not numerical replay results.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Expected result mapping: {location}")
        for key, value in expected.items():
            if key in {"generated_at", "protocol", "split_audit"}:
                continue
            if key not in actual:
                raise ValueError(f"Missing result: {location}/{key}")
            assert_report_matches(actual[key], value, location + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Result list mismatch: {location}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_report_matches(left, right, location + f"/{index}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not isfinite(actual)
            or not isfinite(expected)
            or abs(actual - expected) > 1e-10
        ):
            raise ValueError(f"Metric mismatch: {location}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Result mismatch: {location}")


def rank_metrics(records: dict, metric: str) -> None:
    """Competition ranks, retaining all numerical ties at 1e-12 precision."""
    ordered = sorted(records, key=lambda name: (-records[name][metric]["mean"], name))
    prior = None
    rank = 0
    for position, name in enumerate(ordered, 1):
        value = records[name][metric]["mean"]
        if prior is None or not np.isclose(value, prior, rtol=0, atol=1e-12):
            rank = position
            prior = value
        records[name][metric + "_rank"] = rank


def build_report_results(runs: dict, *, data_revision: str) -> dict:
    """Require all three scenarios and every method before exporting any headline.

    ``runs[scenario][method][seed]`` is the public replay/fit result, including
    its sanitized prediction hash. Never substitute old feature runs when a
    scenario is incomplete. Split dispersion is descriptive, not a fitting SE.
    """
    if set(runs) != set(SCENARIO_SEEDS):
        raise ValueError("A report release requires all three scenarios")
    result = {
        "schema_version": "routestressbench-report-v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "data_revision": data_revision,
            "ranked_implementations": 13,
            "baseline_implementations": 12,
            "budget_points": 100,
            "auc_axis": "common nominal per-request cost budget",
            "auc_envelope": "cumulative maximum router quality over increasing budgets",
            "fixed_model_references_in_router_curve": False,
            "test_population": "official test split; all requests are scored",
            "validation_test_source_group_overlap": 0,
            "multimodal_seeds": list(SCENARIO_SEEDS["multimodal"]),
            "text_and_chinese_seeds": [20260917],
            "run_aggregation": "pointwise mean budget curve before curve-level metrics",
            "curve_metric_uncertainty": (
                "exact nonparametric bootstrap standard error over run curves"
            ),
            "curve_metric_bootstrap": "all 27 ordered resamples of three Multimodal run curves",
            "target_cost_estimator": "minimum realized mean cost on the pointwise mean curve",
            "target_reach_count": "number of individual runs reaching the target",
            "macro_std_identity": "descriptive sample standard deviation across six split scores",
            "alias_audit_scope": "historical adapter audit; only 12 canonical baselines are ranked",
        },
        "macro": {},
        "scenarios": {},
    }
    for scenario, seeds in SCENARIO_SEEDS.items():
        if set(runs[scenario]) != set(METHOD_NAMES):
            raise ValueError(f"Incomplete methods: {scenario}")
        output = {
            "cost_unit": "CNY" if scenario == "chinese" else "USD",
            "model_count": MODEL_COUNTS[scenario],
            "methods": {},
        }
        expected_splits = None
        feature_identity = None
        for method, name in METHOD_NAMES.items():
            cards = runs[scenario][method]
            if {int(seed) for seed in cards} != set(seeds):
                raise ValueError(f"Incomplete fitted runs: {scenario}/{method}")
            cards = {int(seed): card for seed, card in cards.items()}
            if scenario == "multimodal":
                from benchmark.data.feature_provenance import validate_multimodal_identity

                for card in cards.values():
                    identity = card.get("feature_provenance", {})
                    dimension = validate_multimodal_identity(identity)
                    if card.get("feature_dimension") != dimension:
                        raise ValueError("Multimodal feature width and provenance differ")
                    if feature_identity is not None and identity != feature_identity:
                        raise ValueError("Multimodal methods use inconsistent shared features")
                    feature_identity = identity
            split_names = set(cards[seeds[0]]["splits"])
            if expected_splits is None:
                expected_splits = split_names
            if split_names != expected_splits or not {"standard", "robust"} <= split_names:
                raise ValueError(f"Inconsistent reporting populations: {scenario}/{method}")
            if any(set(card["splits"]) != split_names for card in cards.values()):
                raise ValueError("Fitted runs use different reporting populations")
            splits = {
                split: aggregate_report_runs([cards[seed]["splits"][split] for seed in seeds])
                for split in sorted(split_names)
            }
            for split, count in zip(("standard", "robust"), FINAL_COUNTS[scenario], strict=True):
                if splits[split]["query_count"] != count:
                    raise ValueError(f"Unexpected final-test count: {scenario}/{method}/{split}")
            output["methods"][name] = {
                "method_id": method,
                "prediction_sha256": {
                    str(seed): cards[seed]["prediction_sha256"] for seed in seeds
                },
                "splits": splits,
            }
            audit = cards[seeds[0]]["split_audit"]
            if "split_audit" in output and output["split_audit"] != audit:
                raise ValueError("Methods use different test or validation populations")
            output["split_audit"] = audit
        for split in expected_splits:
            rows = {name: method["splits"][split] for name, method in output["methods"].items()}
            for metric in ("normalized_auc", "peak_performance"):
                rank_metrics(rows, metric)
        result["scenarios"][scenario] = output
        if feature_identity is not None:
            output["feature_provenance"] = feature_identity
    for name in METHOD_NAMES.values():
        values = [
            result["scenarios"][scenario]["methods"][name]["splits"][split]["normalized_auc"][
                "mean"
            ]
            for scenario in SCENARIO_SEEDS
            for split in ("standard", "robust")
        ]
        result["macro"][name] = {
            "normalized_auc": {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
        }
    rank_metrics(result["macro"], "normalized_auc")
    return result
