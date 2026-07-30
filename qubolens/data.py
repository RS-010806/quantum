"""Dataset parsing, preprocessing, and deterministic built-in demonstrations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
import random
import re
import statistics
from pathlib import Path
from xml.etree import ElementTree
import zipfile


MISSING = {"", "na", "n/a", "null", "none", "nan", "?"}
MAX_UPLOAD_BYTES = 20_000_000
MAX_ANALYSIS_ROWS = 5_000
MAX_RAW_COLUMNS = 100
MAX_MODEL_FEATURES = 40
TEXT_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252")
TEXT_STOP_WORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "this",
    "was",
    "were",
    "with",
    "your",
}


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


@dataclass(frozen=True)
class TabularUpload:
    headers: tuple[str, ...]
    records: tuple[dict[str, str], ...]
    format_label: str
    total_rows: int
    sampled: bool = False


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


def _clean_headers(values: list[object]) -> list[str]:
    headers: list[str] = []
    counts: dict[str, int] = {}
    for index, value in enumerate(values, start=1):
        base = str(value or "").strip() or f"column_{index}"
        counts[base] = counts.get(base, 0) + 1
        headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return headers


def _record_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _sample_records(
    records: object,
    headers: list[str],
    *,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> tuple[list[dict[str, str]], int]:
    sampled: list[dict[str, str]] = []
    generator = random.Random(704)
    total = 0
    for raw_record in records:
        if not isinstance(raw_record, dict):
            continue
        record = {header: _record_value(raw_record.get(header, "")) for header in headers}
        if not any(value for value in record.values()):
            continue
        total += 1
        if len(sampled) < max_rows:
            sampled.append(record)
        else:
            replacement = generator.randrange(total)
            if replacement < max_rows:
                sampled[replacement] = record
    return sampled, total


def _decode_text(content: bytes) -> str:
    for encoding in TEXT_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The text encoding could not be read. Save the file as UTF-8.")


def _parse_delimited(
    content: bytes,
    *,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> TabularUpload:
    text = _decode_text(content)
    sample = text[:32_768]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        delimiter = next(
            (candidate for candidate in ("\t", ",", ";", "|") if candidate in first_line),
            None,
        )
        if delimiter is None:
            raise ValueError(
                "This text file needs a header row and comma, tab, semicolon, or pipe separators."
            )

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header_row: list[str] | None = None
    for row in reader:
        if any(str(value).strip() for value in row):
            header_row = row
            break
    if not header_row:
        raise ValueError("The file does not contain a readable header row.")
    headers = _clean_headers(list(header_row))
    if not 3 <= len(headers) <= MAX_RAW_COLUMNS + 1:
        raise ValueError(
            f"Use 2 to {MAX_RAW_COLUMNS} input columns plus one prediction target."
        )

    def record_stream():
        for row in reader:
            padded = list(row[: len(headers)]) + [""] * max(0, len(headers) - len(row))
            yield {
                header: _record_value(padded[index])
                for index, header in enumerate(headers)
            }

    records, total = _sample_records(record_stream(), headers, max_rows=max_rows)
    return TabularUpload(
        headers=tuple(headers),
        records=tuple(records),
        format_label={
            ",": "CSV",
            "\t": "tab-separated text",
            ";": "semicolon-separated text",
            "|": "pipe-separated text",
        }.get(delimiter, "delimited text"),
        total_rows=total,
        sampled=total > len(records),
    )


def _json_records(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("JSON data must contain a list of row objects.")
    for key in ("records", "rows", "data", "items"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    list_columns = {
        key: value for key, value in payload.items() if isinstance(value, list)
    }
    if list_columns and len(list_columns) == len(payload):
        lengths = {len(value) for value in list_columns.values()}
        if len(lengths) == 1:
            return [
                {key: values[index] for key, values in list_columns.items()}
                for index in range(next(iter(lengths), 0))
            ]
    raise ValueError(
        "JSON data must be a list of row objects or an object containing records."
    )


def _parse_json_upload(
    content: bytes,
    *,
    line_delimited: bool = False,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> TabularUpload:
    text = _decode_text(content)
    try:
        if line_delimited:
            raw_records = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
            records = [item for item in raw_records if isinstance(item, dict)]
            format_label = "JSON Lines"
        else:
            records = _json_records(json.loads(text))
            format_label = "JSON"
    except json.JSONDecodeError as error:
        raise ValueError(f"The JSON could not be read near line {error.lineno}.") from error
    if not records:
        raise ValueError("The JSON file does not contain any row objects.")
    header_order: dict[str, None] = {}
    for record in records[:500]:
        for key in record:
            header_order.setdefault(str(key).strip(), None)
    headers = _clean_headers(list(header_order))
    if not 3 <= len(headers) <= MAX_RAW_COLUMNS + 1:
        raise ValueError(
            f"Use 2 to {MAX_RAW_COLUMNS} input fields plus one prediction target."
        )
    normalized = [
        {
            clean: _record_value(record.get(original, ""))
            for clean, original in zip(headers, header_order)
        }
        for record in records
    ]
    sampled, total = _sample_records(normalized, headers, max_rows=max_rows)
    return TabularUpload(
        headers=tuple(headers),
        records=tuple(sampled),
        format_label=format_label,
        total_rows=total,
        sampled=total > len(sampled),
    )


def _xlsx_column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    result = 0
    for character in letters.upper():
        result = result * 26 + ord(character) - 64
    return max(0, result - 1)


def _parse_xlsx(
    content: bytes,
    *,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> TabularUpload:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as error:
        raise ValueError("The XLSX workbook could not be opened.") from error
    with archive:
        if sum(item.file_size for item in archive.infolist()) > 120_000_000:
            raise ValueError("The expanded XLSX workbook is too large for an interactive run.")
        try:
            workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
            relationships = ElementTree.fromstring(
                archive.read("xl/_rels/workbook.xml.rels")
            )
        except (KeyError, ElementTree.ParseError) as error:
            raise ValueError("The XLSX workbook structure is incomplete.") from error
        relationship_map = {
            relationship.attrib.get("Id", ""): relationship.attrib.get("Target", "")
            for relationship in relationships
        }
        sheet = next(
            (
                node
                for node in workbook.iter()
                if node.tag.endswith("sheet") and node.attrib.get("state", "visible") == "visible"
            ),
            None,
        )
        if sheet is None:
            raise ValueError("The XLSX workbook has no visible worksheet.")
        relationship_id = next(
            (
                value
                for key, value in sheet.attrib.items()
                if key.endswith("}id") or key == "r:id"
            ),
            "",
        )
        sheet_target = relationship_map.get(relationship_id, "")
        if not sheet_target:
            raise ValueError("The first worksheet could not be located.")
        sheet_path = (
            sheet_target.lstrip("/")
            if sheet_target.startswith("/xl/")
            else f"xl/{sheet_target.lstrip('/')}"
        )
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root:
                shared.append(
                    "".join(node.text or "" for node in item.iter() if node.tag.endswith("}t"))
                )
        try:
            sheet_root = ElementTree.fromstring(archive.read(sheet_path))
        except (KeyError, ElementTree.ParseError) as error:
            raise ValueError("The first XLSX worksheet could not be read.") from error

        raw_rows: list[dict[int, str]] = []
        for row in (node for node in sheet_root.iter() if node.tag.endswith("}row")):
            values: dict[int, str] = {}
            for cell in (node for node in row if node.tag.endswith("}c")):
                index = _xlsx_column_index(cell.attrib.get("r", ""))
                cell_type = cell.attrib.get("t", "")
                value_node = next(
                    (node for node in cell.iter() if node.tag.endswith("}v")),
                    None,
                )
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter() if node.tag.endswith("}t")
                    )
                else:
                    value = value_node.text if value_node is not None else ""
                    if cell_type == "s" and value:
                        try:
                            value = shared[int(value)]
                        except (ValueError, IndexError):
                            value = ""
                    elif cell_type == "b":
                        value = "true" if value == "1" else "false"
                values[index] = _record_value(value)
            if values:
                raw_rows.append(values)
        if not raw_rows:
            raise ValueError("The first XLSX worksheet is empty.")
        width = max(raw_rows[0]) + 1
        headers = _clean_headers([raw_rows[0].get(index, "") for index in range(width)])
        if not 3 <= len(headers) <= MAX_RAW_COLUMNS + 1:
            raise ValueError(
                f"Use 2 to {MAX_RAW_COLUMNS} input columns plus one prediction target."
            )
        normalized = [
            {
                header: row.get(index, "")
                for index, header in enumerate(headers)
            }
            for row in raw_rows[1:]
        ]
        sampled, total = _sample_records(normalized, headers, max_rows=max_rows)
        return TabularUpload(
            headers=tuple(headers),
            records=tuple(sampled),
            format_label="Excel workbook",
            total_rows=total,
            sampled=total > len(sampled),
        )


def parse_tabular_upload(
    content: bytes,
    filename: str,
    *,
    max_rows: int = MAX_ANALYSIS_ROWS,
) -> TabularUpload:
    """Read common tabular uploads without retaining them on disk."""

    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The uploaded file is larger than the 20 MB limit.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx" or content.startswith(b"PK\x03\x04"):
        return _parse_xlsx(content, max_rows=max_rows)
    if suffix == ".xls":
        raise ValueError("Legacy .xls files are not supported. Save the sheet as .xlsx or CSV.")
    if suffix in {".jsonl", ".ndjson"}:
        return _parse_json_upload(content, line_delimited=True, max_rows=max_rows)
    if suffix == ".json" or content.lstrip()[:1] in {b"[", b"{"}:
        return _parse_json_upload(content, max_rows=max_rows)
    return _parse_delimited(content, max_rows=max_rows)


def _missing(value: str) -> bool:
    return value.strip().lower() in MISSING


def _display_name(value: str) -> str:
    label = re.sub(r"[_-]+", " ", value).strip()
    return label[:1].upper() + label[1:] if label else "the selected target"


def _as_number(value: str) -> float | None:
    if _missing(value):
        return None
    normalized = value.strip().replace(",", "")
    if normalized.endswith("%"):
        normalized = normalized[:-1]
        scale = 0.01
    else:
        scale = 1.0
    try:
        result = float(normalized) * scale
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def _as_datetime(value: str) -> float | None:
    if _missing(value):
        return None
    candidate = value.strip().replace("Z", "+00:00")
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for pattern in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(candidate, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() / 86_400


def _correlation_strength(values: list[float], target: list[float]) -> float:
    if len(values) != len(target) or not values:
        return 0.0
    mean_x = sum(values) / len(values)
    mean_y = sum(target) / len(target)
    numerator = sum(
        (value - mean_x) * (label - mean_y)
        for value, label in zip(values, target)
    )
    denominator_x = sum((value - mean_x) ** 2 for value in values)
    denominator_y = sum((label - mean_y) ** 2 for label in target)
    denominator = math.sqrt(denominator_x * denominator_y)
    return abs(numerator / denominator) if denominator > 1e-15 else 0.0


def _parse_target(
    records: list[dict[str, str]],
    target_name: str,
    task: str,
) -> tuple[list[dict[str, str]], list[float], str, list[str]]:
    present_records = [
        record for record in records if not _missing(record.get(target_name, ""))
    ]
    removed = len(records) - len(present_records)
    if len(present_records) < 30:
        raise ValueError("The file needs at least 30 rows with a non-empty prediction target.")
    raw = [record[target_name].strip() for record in present_records]
    unique = sorted(set(raw))
    inferred = "classification" if len(unique) == 2 else "regression"
    selected = inferred if task == "auto" else task
    if selected not in {"classification", "regression"}:
        raise ValueError("Task must be auto, classification, or regression.")
    notes = []
    if removed:
        notes.append(f"Removed {removed:,} rows with a missing prediction target.")
    if selected == "classification":
        if len(unique) != 2:
            raise ValueError(
                "Choose a target with exactly two outcomes, or use a numeric target."
            )
        mapping = {value: float(index) for index, value in enumerate(unique)}
        notes.append(f"Read target outcomes as {unique[0]} and {unique[1]}.")
        return present_records, [mapping[value] for value in raw], selected, notes
    parsed = [_as_number(value) for value in raw]
    if any(value is None for value in parsed):
        raise ValueError(
            "A numeric prediction target is required when there are more than two outcomes."
        )
    numeric_target = [float(value) for value in parsed if value is not None]
    if max(numeric_target) - min(numeric_target) <= 1e-15:
        raise ValueError("The prediction target needs more than one value.")
    return present_records, numeric_target, selected, notes


def _prepare_feature_columns(
    records: list[dict[str, str]],
    raw_features: list[str],
    target: list[float],
    *,
    max_features: int = MAX_MODEL_FEATURES,
) -> tuple[list[str], list[list[float]], list[str]]:
    candidate_names: list[str] = []
    candidate_columns: list[list[float]] = []
    notes: list[str] = []

    def add_candidate(name: str, values: list[float]) -> None:
        if max(values, default=0.0) - min(values, default=0.0) <= 1e-15:
            return
        candidate_names.append(name)
        candidate_columns.append(values)

    for feature in raw_features:
        values = [record.get(feature, "").strip() for record in records]
        present = [value for value in values if not _missing(value)]
        if not present:
            notes.append(f"Skipped empty column: {feature}.")
            continue

        numeric = [_as_number(value) for value in values]
        numeric_present = [value for value in numeric if value is not None]
        if len(numeric_present) / len(present) >= 0.85:
            fill = statistics.median(numeric_present)
            add_candidate(
                feature,
                [fill if value is None else value for value in numeric],
            )
            if len(numeric_present) < len(values):
                notes.append(f"Filled missing or invalid numbers in {feature} with its median.")
            continue

        dates = [_as_datetime(value) for value in values]
        date_present = [value for value in dates if value is not None]
        if len(date_present) / len(present) >= 0.85:
            fill = statistics.median(date_present)
            add_candidate(
                f"{feature} · date",
                [fill if value is None else value for value in dates],
            )
            notes.append(f"Converted {feature} from dates into a time value.")
            continue

        normalized = [
            "<missing>" if _missing(value) else value.strip()
            for value in values
        ]
        frequencies: dict[str, int] = {}
        for value in normalized:
            frequencies[value] = frequencies.get(value, 0) + 1
        unique_count = len(frequencies)
        word_counts = [
            len(re.findall(r"\b[\w'-]+\b", value)) for value in normalized
        ]
        average_words = sum(word_counts) / len(word_counts)
        average_length = sum(len(value) for value in normalized) / len(normalized)
        looks_like_text = average_words >= 4 or average_length >= 48
        category_limit = min(20, max(8, round(math.sqrt(len(values)))))
        if unique_count <= category_limit and not looks_like_text:
            ordered = sorted(
                frequencies,
                key=lambda value: (-frequencies[value], value),
            )
            if unique_count == 2:
                categories = ordered[1:]
            else:
                categories = ordered[:7]
            for category in categories:
                add_candidate(
                    f"{feature} = {category}",
                    [1.0 if value == category else 0.0 for value in normalized],
                )
            if unique_count > len(categories) and unique_count > 2:
                kept = set(categories)
                add_candidate(
                    f"{feature} = other",
                    [0.0 if value in kept else 1.0 for value in normalized],
                )
            notes.append(
                f"Expanded {feature} into "
                f"{min(unique_count, len(categories) + 1)} category indicators."
            )
            continue

        if (
            average_words < 1.4
            and average_length < 24
            and unique_count / len(values) > 0.7
        ):
            notes.append(f"Skipped likely identifier column: {feature}.")
            continue
        char_counts = [float(len(value)) for value in normalized]
        unique_ratios: list[float] = []
        digit_ratios: list[float] = []
        document_tokens: list[set[str]] = []
        for value in normalized:
            words = re.findall(r"\b[\w'-]+\b", value.lower())
            unique_ratios.append(len(set(words)) / len(words) if words else 0.0)
            digit_ratios.append(
                sum(character.isdigit() for character in value) / len(value)
                if value
                else 0.0
            )
            document_tokens.append(
                {
                    word
                    for word in words
                    if len(word) >= 3
                    and word not in TEXT_STOP_WORDS
                    and not word.isdigit()
                }
            )
        add_candidate(f"{feature} · text length", char_counts)
        add_candidate(
            f"{feature} · word count",
            [float(value) for value in word_counts],
        )
        add_candidate(f"{feature} · vocabulary variety", unique_ratios)
        add_candidate(f"{feature} · digit share", digit_ratios)
        document_frequency: dict[str, int] = {}
        for tokens in document_tokens:
            for token in tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        useful_tokens = [
            token
            for token, frequency in document_frequency.items()
            if frequency >= max(3, round(len(values) * 0.02))
            and frequency <= round(len(values) * 0.9)
        ]
        useful_tokens.sort(
            key=lambda token: (
                abs(document_frequency[token] / len(values) - 0.5),
                -document_frequency[token],
                token,
            )
        )
        for token in useful_tokens[:6]:
            add_candidate(
                f'{feature} · contains "{token}"',
                [1.0 if token in tokens else 0.0 for tokens in document_tokens],
            )
        notes.append(
            f"Converted free text in {feature} into structural measures and "
            "common-word indicators."
        )

    if len(candidate_names) < 2:
        raise ValueError(
            "At least two usable inputs are required after empty, constant, and "
            "identifier columns are removed."
        )
    if len(candidate_names) > max_features:
        ranked = sorted(
            range(len(candidate_names)),
            key=lambda index: (
                -_correlation_strength(candidate_columns[index], target),
                candidate_names[index],
            ),
        )[:max_features]
        ranked_set = set(ranked)
        candidate_names = [
            name for index, name in enumerate(candidate_names) if index in ranked_set
        ]
        candidate_columns = [
            column
            for index, column in enumerate(candidate_columns)
            if index in ranked_set
        ]
        notes.append(
            f"Prepared more than {max_features} inputs and kept the "
            f"{max_features} strongest target links for the interactive run."
        )
    return candidate_names, candidate_columns, notes


def _dataset_from_upload(
    upload: TabularUpload,
    *,
    target_name: str,
    task: str = "auto",
    name: str = "Uploaded data",
    max_features: int = MAX_MODEL_FEATURES,
) -> Dataset:
    if target_name not in upload.headers:
        raise ValueError(f"Prediction target '{target_name}' was not found.")
    raw_features = [header for header in upload.headers if header != target_name]
    if len(raw_features) < 2:
        raise ValueError("Choose data with at least two input columns.")
    records, target, selected_task, target_notes = _parse_target(
        list(upload.records), target_name, task
    )
    feature_names, columns, feature_notes = _prepare_feature_columns(
        records,
        raw_features,
        target,
        max_features=max_features,
    )
    rows = tuple(
        tuple(column[index] for column in columns)
        for index in range(len(records))
    )
    notes = target_notes + feature_notes
    if upload.sampled:
        notes.insert(
            0,
            f"Analyzed a repeatable sample of {len(upload.records):,} from "
            f"{upload.total_rows:,} rows.",
        )
    description = (
        f"Your {upload.format_label} file has {upload.total_rows:,} data rows. "
        f"QUBOLens prepared {len(feature_names)} usable inputs"
    )
    if upload.sampled:
        description += f" from a repeatable {len(records):,}-row sample."
    else:
        description += f" from all {len(records):,} rows."
    return Dataset(
        name=name,
        feature_names=tuple(feature_names),
        rows=rows,
        target=tuple(target),
        target_name=target_name,
        task=selected_task,
        notes=tuple(notes),
        question=f"Which inputs best predict {_display_name(target_name)}?",
        description=description,
        target_description=(
            "A yes-or-no outcome"
            if selected_task == "classification"
            else "A numeric outcome"
        ),
    )


def inspect_tabular_upload(
    content: bytes,
    filename: str,
    *,
    target_name: str = "",
) -> dict[str, object]:
    upload = parse_tabular_upload(content, filename)
    selected_target = target_name if target_name in upload.headers else upload.headers[-1]
    result: dict[str, object] = {
        "format": upload.format_label,
        "rows": upload.total_rows,
        "sample_rows": len(upload.records),
        "sampled": upload.sampled,
        "columns": list(upload.headers),
        "target": selected_target,
        "prepared_features": min(MAX_MODEL_FEATURES, len(upload.headers) - 1),
        "task": "unknown",
        "notes": [],
    }
    try:
        dataset = _dataset_from_upload(
            upload,
            target_name=selected_target,
            name=Path(filename).stem or "Uploaded data",
        )
    except ValueError as error:
        result["target_error"] = str(error)
    else:
        result["prepared_features"] = dataset.n_features
        result["task"] = dataset.task
        result["notes"] = list(dataset.notes)
        result["description"] = dataset.description
    return result


def load_tabular_dataset(
    content: bytes,
    *,
    filename: str,
    target_name: str,
    task: str = "auto",
    name: str = "Uploaded data",
    max_rows: int = MAX_ANALYSIS_ROWS,
    max_features: int = MAX_MODEL_FEATURES,
) -> Dataset:
    upload = parse_tabular_upload(content, filename, max_rows=max_rows)
    return _dataset_from_upload(
        upload,
        target_name=target_name,
        task=task,
        name=name,
        max_features=max_features,
    )


def load_csv_dataset(
    text: str,
    *,
    target_name: str,
    task: str = "auto",
    name: str = "Uploaded CSV",
    max_rows: int = MAX_ANALYSIS_ROWS,
    max_features: int = MAX_MODEL_FEATURES,
) -> Dataset:
    """Backward-compatible CSV loader used by the Python API and CLI."""

    return load_tabular_dataset(
        text.encode("utf-8"),
        filename=f"{name}.csv",
        target_name=target_name,
        task=task,
        name=name,
        max_rows=max_rows,
        max_features=max_features,
    )
