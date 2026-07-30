"""A small production-friendly HTTP server for the QUBOLens workbench."""

from __future__ import annotations

import base64
import binascii
from collections import OrderedDict
import copy
import csv
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
import os
from pathlib import Path
import threading
import traceback
from urllib.parse import unquote, urlparse

from .data import (
    Dataset,
    MAX_UPLOAD_BYTES,
    inspect_tabular_upload,
    load_csv_dataset,
    load_tabular_dataset,
    make_demo,
)
from .pipeline import optimize_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_BODY_BYTES = 28_000_000
RESULT_CACHE_SIZE = 24
_RESULT_CACHE: OrderedDict[str, dict[str, object]] = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()


def _cache_key(
    source_fingerprint: str,
    payload: dict[str, object],
) -> str:
    settings = {
        "source": source_fingerprint,
        "target": str(payload.get("target", "")).strip(),
        "task": str(payload.get("task", "auto")),
        "name": str(payload.get("name", ""))[:80],
        "k": int(payload.get("k", 6)),
        "redundancy_weight": float(payload.get("redundancy_weight", 0.65)),
        "quality": str(payload.get("quality", "balanced")),
        "seed": int(payload.get("seed", 42)),
    }
    encoded = json.dumps(
        settings,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cached_result(key: str) -> dict[str, object] | None:
    with _RESULT_CACHE_LOCK:
        result = _RESULT_CACHE.get(key)
        if result is None:
            return None
        _RESULT_CACHE.move_to_end(key)
        copied = copy.deepcopy(result)
    runtime = copied.get("runtime")
    if isinstance(runtime, dict):
        runtime["cache_hit"] = True
    return copied


def _store_result(key: str, result: dict[str, object]) -> None:
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE[key] = copy.deepcopy(result)
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > RESULT_CACHE_SIZE:
            _RESULT_CACHE.popitem(last=False)


class QUBOLensHandler(BaseHTTPRequestHandler):
    server_version = "QUBOLens/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"[qubolens] {self.address_string()} - {format % args}")

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; font-src 'self'; frame-ancestors 'none'",
        )

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length header.") from error
        if length <= 0:
            raise ValueError("Request body is required.")
        if length > MAX_BODY_BYTES:
            raise ValueError("Request is larger than the 28 MB API limit.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Request body must be valid UTF-8 JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json({"status": "ok", "service": "qubolens", "version": "1.0.0"})
            return
        if path == "/api/datasets":
            self._json(
                {
                    "datasets": [
                        {
                            "slug": "edge-failure",
                            "name": "Device failure risk",
                            "question": "Will this device fail within the next 24 hours?",
                            "task": "classification",
                            "samples": 720,
                            "features": 18,
                            "default_k": 6,
                            "download_url": "/api/datasets/edge-failure.csv",
                        },
                        {
                            "slug": "cloud-cost",
                            "name": "Cloud workload cost",
                            "question": "What drives this workload's hourly cloud cost?",
                            "task": "regression",
                            "samples": 680,
                            "features": 16,
                            "default_k": 6,
                            "download_url": "/api/datasets/cloud-cost.csv",
                        },
                    ]
                }
            )
            return
        if path.startswith("/api/datasets/"):
            requested = path.removeprefix("/api/datasets/")
            wants_csv = requested.endswith(".csv")
            slug = requested.removesuffix(".csv")
            if slug not in {"edge-failure", "cloud-cost"}:
                self._json({"error": "Dataset not found."}, HTTPStatus.NOT_FOUND)
                return
            dataset = make_demo(slug)
            if wants_csv:
                self._demo_csv(dataset, slug)
            else:
                self._json(
                    {
                        "slug": slug,
                        "name": dataset.name,
                        "question": dataset.question,
                        "description": dataset.description,
                        "target": dataset.target_name,
                        "target_description": dataset.target_description,
                        "task": dataset.task,
                        "samples": dataset.n_samples,
                        "features": dataset.n_features,
                        "feature_names": list(dataset.feature_names),
                        "notes": list(dataset.notes),
                        "preview": [
                            {
                                **{
                                    name: round(value, 4)
                                    for name, value in zip(
                                        dataset.feature_names, row
                                    )
                                },
                                dataset.target_name: (
                                    int(dataset.target[index])
                                    if dataset.task == "classification"
                                    else round(dataset.target[index], 4)
                                ),
                            }
                            for index, row in enumerate(dataset.rows[:4])
                        ],
                        "download_url": f"/api/datasets/{slug}.csv",
                    }
                )
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/inspect", "/api/optimize"}:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            if path == "/api/inspect":
                content, filename = self._upload_content(payload)
                self._json(
                    inspect_tabular_upload(
                        content,
                        filename,
                        target_name=str(payload.get("target", "")).strip(),
                    )
                )
                return
            source = str(payload.get("source", "demo"))
            if source == "upload":
                content, filename = self._upload_content(payload)
                source_fingerprint = (
                    "upload:" + hashlib.sha256(content).hexdigest()
                )
                target = str(payload.get("target", "")).strip()
                if not target:
                    raise ValueError("Choose the column you want to predict.")
                dataset = load_tabular_dataset(
                    content,
                    filename=filename,
                    target_name=target,
                    task=str(payload.get("task", "auto")),
                    name=str(payload.get("name", "Uploaded data"))[:80],
                )
            elif source == "csv":
                csv_text = payload.get("csv")
                target = str(payload.get("target", "")).strip()
                if not isinstance(csv_text, str) or not target:
                    raise ValueError("CSV text and target column are required.")
                source_fingerprint = (
                    "csv:"
                    + hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
                )
                dataset = load_csv_dataset(
                    csv_text,
                    target_name=target,
                    task=str(payload.get("task", "auto")),
                    name=str(payload.get("name", "Uploaded CSV"))[:80],
                )
            elif source == "demo":
                dataset_slug = str(payload.get("dataset", "edge-failure"))
                source_fingerprint = f"demo:{dataset_slug}"
                dataset = make_demo(dataset_slug)
            else:
                raise ValueError("source must be demo, upload, or csv.")
            cache_key = _cache_key(source_fingerprint, payload)
            cached = _cached_result(cache_key)
            if cached is not None:
                self._json(cached)
                return
            result = optimize_dataset(
                dataset,
                k=int(payload.get("k", 6)),
                redundancy_weight=float(payload.get("redundancy_weight", 0.65)),
                quality=str(payload.get("quality", "balanced")),
                seed=int(payload.get("seed", 42)),
            )
            runtime = result.get("runtime")
            if isinstance(runtime, dict):
                runtime["cache_hit"] = False
            _store_result(cache_key, result)
            self._json(result)
        except (TypeError, ValueError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self._json(
                {"error": "The experiment failed unexpectedly."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _upload_content(payload: dict[str, object]) -> tuple[bytes, str]:
        encoded = payload.get("file")
        filename = str(payload.get("filename", "uploaded-data.csv"))[:160]
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Choose a data file first.")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("The uploaded file could not be decoded.") from error
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("The uploaded file is larger than the 20 MB limit.")
        return content, filename

    def _demo_csv(self, dataset: Dataset, slug: str) -> None:
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow([*dataset.feature_names, dataset.target_name])
        for index, row in enumerate(dataset.rows):
            target = dataset.target[index]
            writer.writerow(
                [
                    *(f"{value:.10g}" for value in row),
                    (
                        int(target)
                        if dataset.task == "classification"
                        else f"{target:.10g}"
                    ),
                ]
            )
        body = stream.getvalue().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="qubolens-{slug}.csv"',
        )
        self.send_header("Cache-Control", "public, max-age=3600")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, requested_path: str) -> None:
        relative = unquote(requested_path).lstrip("/") or "index.html"
        target = (WEB_ROOT / relative).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            target = WEB_ROOT / "index.html"
        body = target.read_bytes()
        mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"{mime_type}; charset=utf-8" if mime_type.startswith("text/") else mime_type,
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Cache-Control",
            (
                "no-cache"
                if target.suffix in {".html", ".css", ".js"}
                else "public, max-age=3600"
            ),
        )
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), QUBOLensHandler)
    print(f"QUBOLens ready at http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
