"""End-to-end feature-budget experiment orchestration."""

from __future__ import annotations

import math
import time

from .core import build_feature_qubo, feature_statistics, solve_qubo
from .data import Dataset
from .evaluate import evaluate_subset


QUALITY = {
    "fast": (10, 110),
    "balanced": (22, 210),
    "deep": (40, 340),
}


def _round_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: round(value, 6) if isinstance(value, float) else value
        for key, value in metrics.items()
    }


def _average_pair_redundancy(
    indices: list[int], redundancy: list[list[float]]
) -> float:
    values = [
        redundancy[i][j]
        for position, i in enumerate(indices)
        for j in indices[position + 1 :]
    ]
    return sum(values) / len(values) if values else 0.0


def _compact_integer(value: int) -> str:
    if value < 1_000_000:
        return f"{value:,}"
    exponent = int(math.log10(value))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f} × 10^{exponent}"


def _frontier_k_values(n_features: int, chosen_k: int) -> list[int]:
    upper = min(n_features, 12)
    if upper <= 7:
        return list(range(1, upper + 1))
    values = {1, 2, chosen_k, upper}
    for fraction in (0.25, 0.45, 0.65, 0.82):
        values.add(max(1, round(upper * fraction)))
    return sorted(value for value in values if value <= upper)


def optimize_dataset(
    dataset: Dataset,
    *,
    k: int,
    redundancy_weight: float = 0.65,
    quality: str = "balanced",
    seed: int = 42,
) -> dict[str, object]:
    """Run the complete QUBO selection, baseline, and diagnostic evaluation."""

    if dataset.n_features > 40:
        raise ValueError("Interactive runs support at most 40 features.")
    if quality not in QUALITY:
        raise ValueError("quality must be fast, balanced, or deep.")
    if not 1 <= k <= dataset.n_features:
        raise ValueError(f"Feature budget must be between 1 and {dataset.n_features}.")
    if not 0.0 <= redundancy_weight <= 2.0:
        raise ValueError("Redundancy weight must be between 0 and 2.")

    started = time.perf_counter()
    rows = [list(row) for row in dataset.rows]
    target = list(dataset.target)
    relevance, redundancy = feature_statistics(rows, target)
    reads, sweeps = QUALITY[quality]
    model = build_feature_qubo(
        relevance,
        redundancy,
        k,
        redundancy_weight=redundancy_weight,
    )
    annealed = solve_qubo(model, reads=reads, sweeps=sweeps, seed=seed)
    selected = [index for index, bit in enumerate(annealed.bits) if bit]
    greedy = sorted(range(dataset.n_features), key=lambda i: relevance[i], reverse=True)[
        :k
    ]
    all_features = list(range(dataset.n_features))

    selected_metrics = evaluate_subset(dataset, selected, folds=5, seed=seed + 49)
    greedy_metrics = evaluate_subset(dataset, greedy, folds=5, seed=seed + 49)
    all_metrics = evaluate_subset(dataset, all_features, folds=5, seed=seed + 49)

    frontier: list[dict[str, object]] = []
    for frontier_k in _frontier_k_values(dataset.n_features, k):
        if frontier_k == k:
            q_indices = selected
            q_metrics = selected_metrics
            greedy_indices = greedy
            g_metrics = greedy_metrics
        else:
            frontier_model = build_feature_qubo(
                relevance,
                redundancy,
                frontier_k,
                redundancy_weight=redundancy_weight,
            )
            frontier_result = solve_qubo(
                frontier_model,
                reads=max(7, reads // 3),
                sweeps=max(80, sweeps // 2),
                seed=seed + frontier_k * 13,
            )
            q_indices = [
                index for index, bit in enumerate(frontier_result.bits) if bit
            ]
            greedy_indices = sorted(
                range(dataset.n_features), key=lambda i: relevance[i], reverse=True
            )[:frontier_k]
            q_metrics = evaluate_subset(dataset, q_indices, folds=5, seed=seed + 49)
            g_metrics = evaluate_subset(
                dataset, greedy_indices, folds=5, seed=seed + 49
            )
        for method, indices, metrics in (
            ("QUBO", q_indices, q_metrics),
            ("Top relevance", greedy_indices, g_metrics),
        ):
            frontier.append(
                {
                    "k": frontier_k,
                    "method": method,
                    "score": round(float(metrics["score"]), 6),
                    "reduction": round(
                        100 * (1 - frontier_k / dataset.n_features), 2
                    ),
                    "redundancy": round(
                        _average_pair_redundancy(indices, redundancy), 6
                    ),
                }
            )

    selected_set = set(selected)
    greedy_set = set(greedy)
    feature_rows = []
    for index, name in enumerate(dataset.feature_names):
        peers = [
            redundancy[index][other]
            for other in selected
            if other != index
        ]
        feature_rows.append(
            {
                "name": name,
                "selected": index in selected_set,
                "greedy": index in greedy_set,
                "relevance": round(relevance[index], 6),
                "selected_redundancy": round(
                    sum(peers) / len(peers) if peers else 0.0, 6
                ),
            }
        )
    feature_rows.sort(key=lambda item: (-int(item["selected"]), -item["relevance"]))

    qubo_redundancy = _average_pair_redundancy(selected, redundancy)
    greedy_redundancy = _average_pair_redundancy(greedy, redundancy)
    score_delta = float(selected_metrics["score"]) - float(greedy_metrics["score"])
    reduction = 100 * (1 - k / dataset.n_features)
    elapsed_ms = (time.perf_counter() - started) * 1000
    score_explanation = (
        "ROC AUC checks whether failing devices rank above healthy ones. "
        "0.50 is random ordering; 1.00 is perfect ordering."
        if dataset.task == "classification"
        else "R² checks how much of the cost variation the model explains. "
        "1.00 is perfect; 0.00 is no better than always predicting the average."
    )

    if score_delta >= 0.002 and qubo_redundancy < greedy_redundancy:
        finding = (
            "QUBOLens found a feature set with less overlap and a slightly stronger "
            f"{selected_metrics['score_label']} than the simple ranking."
        )
    elif qubo_redundancy + 0.01 < greedy_redundancy:
        finding = (
            "QUBOLens removed overlapping inputs while keeping model quality nearly "
            "the same."
        )
    else:
        finding = (
            "At this feature limit, both methods found a very similar result. That is "
            "useful too: the simpler option may be enough."
        )

    return {
        "dataset": {
            "name": dataset.name,
            "target": dataset.target_name,
            "task": dataset.task,
            "samples": dataset.n_samples,
            "features": dataset.n_features,
            "notes": list(dataset.notes),
            "question": dataset.question or f"Which inputs best predict {dataset.target_name}?",
            "description": dataset.description,
            "target_description": dataset.target_description,
        },
        "selection": {
            "k": k,
            "names": [dataset.feature_names[index] for index in selected],
            "indices": selected,
            "energy": round(annealed.energy, 6),
            "feature_reduction": round(reduction, 2),
            "search_space": math.comb(dataset.n_features, k),
            "search_space_label": _compact_integer(math.comb(dataset.n_features, k)),
            "average_relevance": round(
                sum(relevance[index] for index in selected) / len(selected), 6
            ),
            "average_redundancy": round(qubo_redundancy, 6),
        },
        "benchmark": {
            "qubo": _round_metrics(selected_metrics),
            "greedy": _round_metrics(greedy_metrics),
            "all_features": _round_metrics(all_metrics),
        },
        "insight": {
            "finding": finding,
            "score_delta_vs_greedy": round(score_delta, 6),
            "redundancy_delta_vs_greedy": round(
                qubo_redundancy - greedy_redundancy, 6
            ),
            "score_explanation": score_explanation,
            "premise": (
                "Quantum ideas can be useful before quantum hardware is: the first "
                "step is framing a messy feature choice as a clear optimization "
                "problem that can be tested and compared."
            ),
        },
        "features": feature_rows,
        "frontier": frontier,
        "annealing": {
            "reads": annealed.reads,
            "sweeps": annealed.sweeps,
            "seed": annealed.seed,
            "feasible_reads": annealed.feasible_reads,
            "curve": [
                {"sweep": sweep, "energy": round(energy, 6)}
                for sweep, energy in annealed.trace
            ],
        },
        "qubo": {
            "formula": (
                "E(x) = −Σ rᵢxᵢ + λΣ Rᵢⱼxᵢxⱼ "
                f"+ {model.cardinality_penalty:g}(Σxᵢ − {k})²"
            ),
            "matrix": [
                [round(value, 5) for value in row] for row in model.dense()
            ],
            "export": model.serializable(list(dataset.feature_names)),
            "nonzero_terms": dataset.n_features * (dataset.n_features + 1) // 2,
        },
        "runtime": {
            "total_ms": round(elapsed_ms, 2),
            "environment": "built-in search",
            "dependencies": 0,
        },
        "caveat": (
            "Use these scores to explore a direction, not as a final production "
            "claim. Confirm your choice on data the selector has never seen."
        ),
    }
