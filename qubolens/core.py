"""QUBO construction and a deterministic, dependency-free simulated annealer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(left: list[float], right: list[float]) -> float:
    """Return a numerically safe Pearson correlation."""

    if len(left) != len(right) or not left:
        return 0.0
    mean_left = _mean(left)
    mean_right = _mean(right)
    numerator = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for a, b in zip(left, right):
        da = a - mean_left
        db = b - mean_right
        numerator += da * db
        left_sq += da * da
        right_sq += db * db
    denominator = math.sqrt(left_sq * right_sq)
    return numerator / denominator if denominator > 1e-15 else 0.0


def feature_statistics(
    rows: list[list[float]], target: list[float]
) -> tuple[list[float], list[list[float]]]:
    """Compute normalized target relevance and pairwise redundancy."""

    if not rows or not rows[0]:
        raise ValueError("The dataset must contain at least one feature.")
    columns = [list(column) for column in zip(*rows)]
    raw_relevance = [abs(pearson(column, target)) for column in columns]
    maximum = max(raw_relevance, default=0.0)
    relevance = (
        [value / maximum for value in raw_relevance]
        if maximum > 1e-15
        else [0.0 for _ in raw_relevance]
    )
    n_features = len(columns)
    redundancy = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for i in range(n_features):
        redundancy[i][i] = 1.0
        for j in range(i + 1, n_features):
            value = abs(pearson(columns[i], columns[j]))
            redundancy[i][j] = value
            redundancy[j][i] = value
    return relevance, redundancy


@dataclass(frozen=True)
class QuboModel:
    """Upper-triangular QUBO plus the constant offset omitted by solvers."""

    matrix: tuple[tuple[float, ...], ...]
    offset: float
    k: int
    relevance: tuple[float, ...]
    redundancy: tuple[tuple[float, ...], ...]
    redundancy_weight: float
    cardinality_penalty: float

    @property
    def size(self) -> int:
        return len(self.matrix)

    def energy(self, bits: Iterable[int], include_offset: bool = True) -> float:
        state = list(bits)
        if len(state) != self.size:
            raise ValueError("Bit vector length does not match the QUBO.")
        energy = 0.0
        for i, bit_i in enumerate(state):
            if not bit_i:
                continue
            energy += self.matrix[i][i]
            for j in range(i + 1, self.size):
                if state[j]:
                    energy += self.matrix[i][j]
        return energy + (self.offset if include_offset else 0.0)

    def direct_objective(self, bits: Iterable[int]) -> float:
        """Evaluate the human-readable objective for implementation checks."""

        state = list(bits)
        chosen = [index for index, bit in enumerate(state) if bit]
        relevance_term = -sum(self.relevance[index] for index in chosen)
        scale = self.redundancy_weight / max(1, self.k - 1)
        redundancy_term = sum(
            scale * self.redundancy[i][j]
            for position, i in enumerate(chosen)
            for j in chosen[position + 1 :]
        )
        cardinality_term = self.cardinality_penalty * (len(chosen) - self.k) ** 2
        return relevance_term + redundancy_term + cardinality_term

    def dense(self) -> list[list[float]]:
        dense_matrix = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(i, self.size):
                dense_matrix[i][j] = self.matrix[i][j]
                dense_matrix[j][i] = self.matrix[i][j]
        return dense_matrix

    def serializable(self, names: list[str]) -> dict[str, object]:
        terms = {
            f"{names[i]},{names[j]}": round(self.matrix[i][j], 8)
            for i in range(self.size)
            for j in range(i, self.size)
            if abs(self.matrix[i][j]) > 1e-12
        }
        return {
            "format": "QUBO",
            "variables": names,
            "linear_and_quadratic_terms": terms,
            "offset": round(self.offset, 8),
            "objective": (
                "-relevance + redundancy + "
                f"{self.cardinality_penalty:g} * (sum(x) - {self.k})^2"
            ),
        }


def build_feature_qubo(
    relevance: list[float],
    redundancy: list[list[float]],
    k: int,
    redundancy_weight: float = 0.65,
    cardinality_penalty: float = 2.5,
) -> QuboModel:
    """Encode fixed-cardinality, relevance/redundancy feature selection as QUBO."""

    n_features = len(relevance)
    if n_features == 0:
        raise ValueError("At least one feature is required.")
    if not 1 <= k <= n_features:
        raise ValueError(f"k must be between 1 and {n_features}.")
    if len(redundancy) != n_features or any(
        len(row) != n_features for row in redundancy
    ):
        raise ValueError("Redundancy must be a square feature matrix.")
    if redundancy_weight < 0:
        raise ValueError("Redundancy weight cannot be negative.")
    if cardinality_penalty <= 0:
        raise ValueError("Cardinality penalty must be positive.")

    matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    pair_scale = redundancy_weight / max(1, k - 1)
    for i in range(n_features):
        matrix[i][i] = -relevance[i] + cardinality_penalty * (1 - 2 * k)
        for j in range(i + 1, n_features):
            matrix[i][j] = (
                pair_scale * max(0.0, redundancy[i][j])
                + 2 * cardinality_penalty
            )
    model = QuboModel(
        matrix=tuple(tuple(row) for row in matrix),
        offset=cardinality_penalty * k * k,
        k=k,
        relevance=tuple(relevance),
        redundancy=tuple(tuple(row) for row in redundancy),
        redundancy_weight=redundancy_weight,
        cardinality_penalty=cardinality_penalty,
    )
    return model


def _flip_delta(model: QuboModel, state: list[int], index: int) -> float:
    local_field = model.matrix[index][index]
    for other, selected in enumerate(state):
        if other == index or not selected:
            continue
        i, j = sorted((index, other))
        local_field += model.matrix[i][j]
    return (1 - 2 * state[index]) * local_field


def _repair_cardinality(model: QuboModel, state: list[int]) -> list[int]:
    repaired = state[:]
    while sum(repaired) > model.k:
        candidates = [i for i, bit in enumerate(repaired) if bit]
        index = min(candidates, key=lambda item: _flip_delta(model, repaired, item))
        repaired[index] = 0
    while sum(repaired) < model.k:
        candidates = [i for i, bit in enumerate(repaired) if not bit]
        index = min(candidates, key=lambda item: _flip_delta(model, repaired, item))
        repaired[index] = 1
    return repaired


@dataclass(frozen=True)
class AnnealResult:
    bits: tuple[int, ...]
    energy: float
    trace: tuple[tuple[int, float], ...]
    reads: int
    sweeps: int
    feasible_reads: int
    seed: int


def solve_qubo(
    model: QuboModel,
    *,
    reads: int = 28,
    sweeps: int = 220,
    seed: int = 42,
) -> AnnealResult:
    """Solve a QUBO with Metropolis simulated annealing on the local CPU."""

    if reads < 1 or sweeps < 1:
        raise ValueError("reads and sweeps must be positive.")
    n_features = model.size
    max_bias = max(
        abs(model.matrix[i][j])
        for i in range(n_features)
        for j in range(i, n_features)
    )
    start_temperature = max(0.8, max_bias * 0.55)
    end_temperature = 0.008
    best_state: list[int] | None = None
    best_energy = math.inf
    best_trace: list[tuple[int, float]] = []
    feasible_reads = 0
    checkpoint_every = max(1, sweeps // 48)

    for read in range(reads):
        generator = random.Random(seed + read * 104_729)
        selected = set(generator.sample(range(n_features), model.k))
        state = [1 if index in selected else 0 for index in range(n_features)]
        current_energy = model.energy(state, include_offset=False)
        read_best = state[:]
        read_best_energy = current_energy
        trace = [(0, current_energy + model.offset)]

        for sweep in range(1, sweeps + 1):
            fraction = sweep / sweeps
            temperature = start_temperature * (
                end_temperature / start_temperature
            ) ** fraction
            order = list(range(n_features))
            generator.shuffle(order)
            for index in order:
                delta = _flip_delta(model, state, index)
                if delta <= 0 or generator.random() < math.exp(-delta / temperature):
                    state[index] = 1 - state[index]
                    current_energy += delta
                    if sum(state) == model.k and current_energy < read_best_energy:
                        read_best = state[:]
                        read_best_energy = current_energy
            if sweep % checkpoint_every == 0 or sweep == sweeps:
                trace.append((sweep, read_best_energy + model.offset))

        if sum(read_best) != model.k:
            read_best = _repair_cardinality(model, read_best)
            read_best_energy = model.energy(read_best, include_offset=False)
        else:
            feasible_reads += 1
        if read_best_energy < best_energy:
            best_state = read_best
            best_energy = read_best_energy
            best_trace = trace

    if best_state is None:
        raise RuntimeError("The annealer did not produce a solution.")
    return AnnealResult(
        bits=tuple(best_state),
        energy=best_energy + model.offset,
        trace=tuple(best_trace),
        reads=reads,
        sweeps=sweeps,
        feasible_reads=feasible_reads,
        seed=seed,
    )
