"""Dataset parsing, preprocessing, and deterministic built-in demonstrations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import math
import random
import statistics


MISSING = {"", "na", "n/a", "null", "none", "nan", "?"}


@dataclass(frozen=True)
class Dataset:
    name: str
    feature_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    target: tuple[float, ...]
    target_name: str
    task: str
    notes: tuple[str, ...] = ()
    question: str = ""
    description: str = ""
    target_description: str = ""

    @property
    def n_samples(self) -> int:
        return len(self.rows)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _gaussian(generator: random.Random) -> float:
    u1 = max(generator.random(), 1e-12)
    u2 = generator.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def _edge_failure_demo() -> Dataset:
    generator = random.Random(17)
    names = (
        "thermal_gradient",
        "vibration_rms",
        "voltage_drift",
        "packet_jitter",
        "fan_delta",
        "load_spike",
        "thermal_sensor_b",
        "vibration_peak",
        "voltage_backup",
        "latency_p95",
        "ambient_humidity",
        "queue_depth",
        "memory_pressure",
        "disk_wait",
        "rack_position",
        "firmware_age",
        "noise_probe_a",
        "noise_probe_b",
    )
    rows: list[tuple[float, ...]] = []
    target: list[float] = []
    for _ in range(720):
        thermal = _gaussian(generator)
        vibration = _gaussian(generator)
        voltage = _gaussian(generator)
        jitter = _gaussian(generator)
        fan = _gaussian(generator)
        load = _gaussian(generator)
        thermal_b = 0.91 * thermal + 0.25 * _gaussian(generator)
        vibration_peak = 0.88 * vibration + 0.29 * _gaussian(generator)
        voltage_backup = 0.86 * voltage + 0.33 * _gaussian(generator)
        latency = 0.67 * jitter + 0.45 * load + 0.35 * _gaussian(generator)
        ambient = 0.25 * thermal + 0.9 * _gaussian(generator)
        queue = 0.52 * load + 0.74 * _gaussian(generator)
        memory = 0.43 * load + 0.82 * _gaussian(generator)
        disk = 0.25 * load + 0.96 * _gaussian(generator)
        rack = _gaussian(generator)
        firmware = _gaussian(generator)
        noise_a = _gaussian(generator)
        noise_b = _gaussian(generator)
        logit = (
            1.35 * thermal
            + 1.05 * vibration
            + 0.82 * voltage
            + 0.66 * jitter
            - 0.62 * fan
            + 0.5 * load
            + 0.32 * thermal * vibration
            + 0.35 * _gaussian(generator)
            - 0.15
        )
        label = 1.0 if generator.random() < _sigmoid(logit) else 0.0
        rows.append(
            (
                thermal,
                vibration,
                voltage,
                jitter,
                fan,
                load,
                thermal_b,
                vibration_peak,
                voltage_backup,
                latency,
                ambient,
                queue,
                memory,
                disk,
                rack,
                firmware,
                noise_a,
                noise_b,
            )
        )
        target.append(label)
    return Dataset(
        name="Device failure risk",
        feature_names=names,
        rows=tuple(rows),
        target=tuple(target),
        target_name="fails_within_24h",
        task="classification",
        notes=(
            "Synthetic, seeded telemetry with informative, redundant, and noise sensors.",
            "The duplicated sensors make the relevance-versus-redundancy trade-off visible.",
        ),
        question="Will this device fail within the next 24 hours?",
        description=(
            "A guided example using 720 simulated device snapshots and 18 sensor "
            "readings to predict failure within 24 hours."
        ),
        target_description="A yes-or-no failure prediction",
    )


def _cloud_cost_demo() -> Dataset:
    generator = random.Random(29)
    names = (
        "request_rate",
        "payload_kb",
        "cache_miss_rate",
        "cpu_saturation",
        "memory_gb",
        "region_distance",
        "request_rate_lag",
        "cpu_load_copy",
        "network_egress",
        "queue_wait",
        "replica_count",
        "cold_start_rate",
        "storage_iops",
        "weekday",
        "noise_meter_a",
        "noise_meter_b",
    )
    rows: list[tuple[float, ...]] = []
    target: list[float] = []
    for _ in range(680):
        rate = _gaussian(generator)
        payload = _gaussian(generator)
        miss = _gaussian(generator)
        cpu = 0.5 * rate + 0.78 * _gaussian(generator)
        memory = _gaussian(generator)
        distance = _gaussian(generator)
        rate_lag = 0.9 * rate + 0.3 * _gaussian(generator)
        cpu_copy = 0.89 * cpu + 0.31 * _gaussian(generator)
        egress = 0.55 * payload + 0.32 * rate + 0.65 * _gaussian(generator)
        queue = 0.62 * cpu + 0.56 * _gaussian(generator)
        replicas = 0.5 * rate + 0.7 * _gaussian(generator)
        cold = 0.38 * miss + 0.84 * _gaussian(generator)
        iops = 0.3 * payload + 0.92 * _gaussian(generator)
        weekday = generator.randrange(7) / 3 - 1
        noise_a = _gaussian(generator)
        noise_b = _gaussian(generator)
        cost = (
            21.0
            + 6.8 * rate
            + 4.2 * payload
            + 3.7 * miss
            + 5.1 * cpu
            + 2.4 * memory
            + 1.9 * distance
            + 1.2 * rate * payload
            + 1.8 * _gaussian(generator)
        )
        rows.append(
            (
                rate,
                payload,
                miss,
                cpu,
                memory,
                distance,
                rate_lag,
                cpu_copy,
                egress,
                queue,
                replicas,
                cold,
                iops,
                weekday,
                noise_a,
                noise_b,
            )
        )
        target.append(cost)
    return Dataset(
        name="Cloud workload cost",
        feature_names=names,
        rows=tuple(rows),
        target=tuple(target),
        target_name="hourly_cost_usd",
        task="regression",
        notes=(
            "Synthetic, seeded service telemetry for a cost-regression pipeline.",
            "Proxy metrics deliberately overlap so a non-redundant subset is useful.",
        ),
        question="What drives this workload's hourly cloud cost?",
        description=(
            "A guided example using 680 simulated hourly workloads and 16 service "
            "signals to estimate cost in US dollars."
        ),
        target_description="Estimated hourly cost in US dollars",
    )


def make_demo(slug: str = "edge-failure") -> Dataset:
    demos = {
        "edge-failure": _edge_failure_demo,
        "cloud-cost": _cloud_cost_demo,
    }
    try:
        return demos[slug]()
    except KeyError as error:
        raise ValueError(f"Unknown demo dataset: {slug}") from error


def _parse_numeric(values: list[str]) -> tuple[list[float], str]:
    present = [value.strip() for value in values if value.strip().lower() not in MISSING]
    numeric = True
    parsed: list[float] = []
    for value in present:
        try:
            parsed.append(float(value))
        except ValueError:
            numeric = False
            break
    if numeric:
        fill = statistics.median(parsed) if parsed else 0.0
        result = [
            fill if value.strip().lower() in MISSING else float(value)
            for value in values
        ]
        return result, "numeric"

    categories = sorted(set(present))
    mapping = {value: float(index) for index, value in enumerate(categories)}
    mode = statistics.mode(present) if present else ""
    result = [
        mapping.get(mode, 0.0)
        if value.strip().lower() in MISSING
        else mapping[value.strip()]
        for value in values
    ]
    return result, f"categorical ({len(categories)} levels)"


def load_csv_dataset(
    text: str,
    *,
    target_name: str,
    task: str = "auto",
    name: str = "Uploaded CSV",
    max_rows: int = 2500,
    max_features: int = 40,
) -> Dataset:
    """Parse an in-memory CSV, encoding categories without external libraries."""

    if len(text.encode("utf-8")) > 2_500_000:
        raise ValueError("CSV is larger than the 2.5 MB in-memory limit.")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("CSV must include a header row.")
    headers = [header.strip() for header in reader.fieldnames]
    if target_name not in headers:
        raise ValueError(f"Target column '{target_name}' was not found.")
    feature_names = [header for header in headers if header != target_name]
    if not 2 <= len(feature_names) <= max_features:
        raise ValueError(f"CSV must contain 2 to {max_features} feature columns.")

    records: list[dict[str, str]] = []
    for row in reader:
        if len(records) >= max_rows:
            break
        records.append({key.strip(): (value or "") for key, value in row.items() if key})
    if len(records) < 30:
        raise ValueError("CSV must contain at least 30 data rows.")

    notes: list[str] = []
    columns: list[list[float]] = []
    kept_features: list[str] = []
    for feature in feature_names:
        values, kind = _parse_numeric([record.get(feature, "") for record in records])
        if max(values, default=0.0) - min(values, default=0.0) <= 1e-15:
            notes.append(f"Dropped constant feature: {feature}.")
            continue
        columns.append(values)
        kept_features.append(feature)
        if kind != "numeric":
            notes.append(f"Ordinal-encoded {feature}: {kind}.")
    if len(kept_features) < 2:
        raise ValueError("At least two non-constant features are required.")

    target_values = [record.get(target_name, "").strip() for record in records]
    present_targets = [value for value in target_values if value.lower() not in MISSING]
    unique_targets = sorted(set(present_targets))
    inferred_task = "classification" if len(unique_targets) == 2 else "regression"
    selected_task = inferred_task if task == "auto" else task
    if selected_task not in {"classification", "regression"}:
        raise ValueError("Task must be auto, classification, or regression.")

    if selected_task == "classification":
        if len(unique_targets) != 2:
            raise ValueError("This version supports binary classification targets.")
        mapping = {value: float(index) for index, value in enumerate(unique_targets)}
        fill = statistics.mode(present_targets)
        target = [
            mapping[fill] if value.lower() in MISSING else mapping[value]
            for value in target_values
        ]
        notes.append(
            f"Mapped target labels to 0/1: {unique_targets[0]} → 0, {unique_targets[1]} → 1."
        )
    else:
        try:
            parsed_target = [float(value) for value in present_targets]
        except ValueError as error:
            raise ValueError("Regression target must be numeric.") from error
        fill_value = statistics.median(parsed_target)
        target = [
            fill_value if value.lower() in MISSING else float(value)
            for value in target_values
        ]

    rows = tuple(tuple(column[row] for column in columns) for row in range(len(records)))
    if len(records) == max_rows:
        notes.append(f"Used the first {max_rows:,} rows for an interactive CPU run.")
    return Dataset(
        name=name,
        feature_names=tuple(kept_features),
        rows=rows,
        target=tuple(target),
        target_name=target_name,
        task=selected_task,
        notes=tuple(notes),
        question=f"Which inputs best predict {target_name}?",
        description=(
            f"Your uploaded dataset contains {len(records):,} rows and "
            f"{len(kept_features)} usable inputs."
        ),
        target_description=(
            "A yes-or-no outcome"
            if selected_task == "classification"
            else "A numeric outcome"
        ),
    )
