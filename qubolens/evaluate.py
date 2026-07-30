"""Small, dependency-free linear models for honest comparative diagnostics."""

from __future__ import annotations

import math
import random
import time

from .data import Dataset


def _sigmoid(value: float) -> float:
    clipped = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _folds(target: list[float], task: str, count: int, seed: int) -> list[list[int]]:
    generator = random.Random(seed)
    buckets = [[] for _ in range(count)]
    if task == "classification":
        groups: dict[float, list[int]] = {}
        for index, value in enumerate(target):
            groups.setdefault(value, []).append(index)
        for group in groups.values():
            generator.shuffle(group)
            for position, index in enumerate(group):
                buckets[position % count].append(index)
    else:
        indices = list(range(len(target)))
        generator.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % count].append(index)
    return buckets


def _prepare_fold(
    rows: list[list[float]],
    train_indices: list[int],
    test_indices: list[int],
    feature_indices: list[int],
) -> tuple[list[list[float]], list[list[float]]]:
    means: list[float] = []
    scales: list[float] = []
    for feature in feature_indices:
        values = [rows[index][feature] for index in train_indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        scales.append(math.sqrt(variance) if variance > 1e-15 else 1.0)

    def transform(indices: list[int]) -> list[list[float]]:
        return [
            [
                (rows[row_index][feature] - means[position]) / scales[position]
                for position, feature in enumerate(feature_indices)
            ]
            for row_index in indices
        ]

    return transform(train_indices), transform(test_indices)


def _fit_logistic(
    features: list[list[float]],
    target: list[float],
    *,
    epochs: int = 150,
) -> list[float]:
    width = len(features[0])
    weights = [0.0] * (width + 1)
    learning_rate = 0.16
    regularization = 0.012
    for epoch in range(max(1, epochs)):
        gradients = [0.0] * (width + 1)
        for row, label in zip(features, target):
            linear = weights[0]
            for column in range(width):
                linear += weights[column + 1] * row[column]
            prediction = _sigmoid(linear)
            error = prediction - label
            gradients[0] += error
            for column in range(width):
                gradients[column + 1] += error * row[column]
        step = learning_rate / (1 + 0.012 * epoch)
        size = len(features)
        weights[0] -= step * gradients[0] / size
        for column in range(1, len(weights)):
            weights[column] -= step * (
                gradients[column] / size + regularization * weights[column]
            )
    return weights


def _fit_linear(
    features: list[list[float]],
    target: list[float],
    *,
    epochs: int = 190,
) -> tuple[list[float], float, float]:
    width = len(features[0])
    mean_target = sum(target) / len(target)
    variance = sum((value - mean_target) ** 2 for value in target) / len(target)
    scale_target = math.sqrt(variance) if variance > 1e-15 else 1.0
    normalized = [(value - mean_target) / scale_target for value in target]
    weights = [0.0] * (width + 1)
    learning_rate = 0.075
    regularization = 0.01
    for epoch in range(max(1, epochs)):
        gradients = [0.0] * (width + 1)
        for row, label in zip(features, normalized):
            prediction = weights[0]
            for column in range(width):
                prediction += weights[column + 1] * row[column]
            error = prediction - label
            gradients[0] += error
            for column in range(width):
                gradients[column + 1] += error * row[column]
        step = learning_rate / (1 + 0.009 * epoch)
        size = len(features)
        weights[0] -= step * gradients[0] / size
        for column in range(1, len(weights)):
            weights[column] -= step * (
                gradients[column] / size + regularization * weights[column]
            )
    return weights, mean_target, scale_target


def _evaluation_indices(
    target: list[float],
    task: str,
    max_samples: int | None,
    seed: int,
) -> list[int]:
    """Choose a repeatable, representative validation sample."""

    sample_count = len(target)
    if max_samples is None or max_samples <= 0 or sample_count <= max_samples:
        return list(range(sample_count))

    generator = random.Random(seed)
    if task != "classification":
        indices = list(range(sample_count))
        generator.shuffle(indices)
        return sorted(indices[:max_samples])

    groups: dict[float, list[int]] = {}
    for index, label in enumerate(target):
        groups.setdefault(label, []).append(index)
    for indices in groups.values():
        generator.shuffle(indices)

    exact = {
        label: max_samples * len(indices) / sample_count
        for label, indices in groups.items()
    }
    quotas = {
        label: min(len(indices), max(1, math.floor(exact[label])))
        for label, indices in groups.items()
    }

    while sum(quotas.values()) < max_samples:
        candidates = [
            label for label, indices in groups.items() if quotas[label] < len(indices)
        ]
        if not candidates:
            break
        label = max(
            candidates,
            key=lambda value: (exact[value] - quotas[value], len(groups[value])),
        )
        quotas[label] += 1

    while sum(quotas.values()) > max_samples:
        candidates = [label for label in groups if quotas[label] > 1]
        if not candidates:
            break
        label = max(
            candidates,
            key=lambda value: (quotas[value] - exact[value], quotas[value]),
        )
        quotas[label] -= 1

    chosen: list[int] = []
    for label, indices in groups.items():
        chosen.extend(indices[: quotas[label]])
    return sorted(chosen)


def _auc(labels: list[float], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_count = sum(label == 1.0 for label in labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return 0.5
    positive_rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            label == 1.0 for _, label in pairs[start:end]
        )
        start = end
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def evaluate_subset(
    dataset: Dataset,
    feature_indices: list[int],
    *,
    folds: int = 5,
    seed: int = 91,
    max_samples: int | None = None,
    training_steps: int | None = None,
) -> dict[str, float | str | int]:
    """Evaluate a fixed subset with post-selection cross-validation."""

    if not feature_indices:
        raise ValueError("At least one feature is required for evaluation.")
    selected_rows = _evaluation_indices(
        dataset.target,
        dataset.task,
        max_samples=max_samples,
        seed=seed,
    )
    rows = [list(dataset.rows[index]) for index in selected_rows]
    target = [dataset.target[index] for index in selected_rows]
    fold_indices = _folds(target, dataset.task, folds, seed)
    observed: list[float] = []
    predicted: list[float] = []
    started = time.perf_counter()

    for test_indices in fold_indices:
        test_set = set(test_indices)
        train_indices = [index for index in range(len(rows)) if index not in test_set]
        train_x, test_x = _prepare_fold(
            rows, train_indices, test_indices, feature_indices
        )
        train_y = [target[index] for index in train_indices]
        test_y = [target[index] for index in test_indices]
        if dataset.task == "classification":
            weights = _fit_logistic(
                train_x,
                train_y,
                epochs=training_steps if training_steps is not None else 150,
            )
            fold_predictions = [
                _sigmoid(
                    weights[0]
                    + sum(
                        weights[column + 1] * row[column]
                        for column in range(len(row))
                    )
                )
                for row in test_x
            ]
        else:
            weights, mean_target, scale_target = _fit_linear(
                train_x,
                train_y,
                epochs=training_steps if training_steps is not None else 190,
            )
            fold_predictions = [
                (
                    weights[0]
                    + sum(
                        weights[column + 1] * row[column]
                        for column in range(len(row))
                    )
                )
                * scale_target
                + mean_target
                for row in test_x
            ]
        observed.extend(test_y)
        predicted.extend(fold_predictions)

    elapsed_ms = (time.perf_counter() - started) * 1000
    if dataset.task == "classification":
        score = _auc(observed, predicted)
        accuracy = sum(
            (probability >= 0.5) == bool(label)
            for label, probability in zip(observed, predicted)
        ) / len(observed)
        clipped = [max(1e-9, min(1 - 1e-9, value)) for value in predicted]
        log_loss = -sum(
            label * math.log(probability)
            + (1 - label) * math.log(1 - probability)
            for label, probability in zip(observed, clipped)
        ) / len(observed)
        return {
            "score": score,
            "score_label": "ROC AUC",
            "secondary": accuracy,
            "secondary_label": "Accuracy",
            "loss": log_loss,
            "loss_label": "Log loss",
            "cv_folds": folds,
            "validation_samples": len(target),
            "fit_ms": elapsed_ms,
            "proxy_ops": 2 * len(feature_indices) + 1,
        }

    mean_observed = sum(observed) / len(observed)
    residual = sum((actual - estimate) ** 2 for actual, estimate in zip(observed, predicted))
    total = sum((actual - mean_observed) ** 2 for actual in observed)
    r_squared = 1 - residual / total if total > 1e-15 else 0.0
    rmse = math.sqrt(residual / len(observed))
    return {
        "score": r_squared,
        "score_label": "R²",
        "secondary": rmse,
        "secondary_label": "RMSE",
        "loss": rmse,
        "loss_label": "RMSE",
        "cv_folds": folds,
        "validation_samples": len(target),
        "fit_ms": elapsed_ms,
        "proxy_ops": 2 * len(feature_indices) + 1,
    }
