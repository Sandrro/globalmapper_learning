"""Utility helpers for running centroid inference through the training script."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field


class InferParams(BaseModel):
    """Parameters that control the inference call."""

    slots: int = Field(..., ge=1, description="Number of slots for inference")
    knn: int = Field(..., ge=1, description="k value for kNN graph construction")
    e_thr: float = Field(..., description="Threshold for edge activation")
    il_thr: float = Field(..., description="Threshold for is_living probability")
    sv1_thr: float = Field(..., description="Threshold for the first service head")


class CentroidRequest(BaseModel):
    """Payload accepted by the centroid generation service."""

    train_script: str
    model_ckpt: str
    zone_attr: str
    zone_label: str
    request_id: Optional[str] = None
    feature: Dict[str, Any]
    infer_params: InferParams
    config: Optional[str] = None
    device: Optional[str] = None
    services_target: Optional[Dict[str, int]] = None
    la_target: Optional[float] = None
    floors_avg: Optional[float] = None
    python_executable: Optional[str] = None


class CentroidResult(BaseModel):
    """Result of the centroid inference request."""

    features: List[Dict[str, Any]]


class CommandLogEntry(BaseModel):
    """Captured stdout/stderr from an inference subprocess invocation."""

    timestamp: datetime
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    success: bool


_COMMAND_LOGS: deque[CommandLogEntry] = deque(maxlen=200)
_COMMAND_LOGS_LOCK = Lock()


def _record_command_log(entry: CommandLogEntry) -> None:
    """Persist a command log entry in the in-memory history."""

    with _COMMAND_LOGS_LOCK:
        _COMMAND_LOGS.append(entry)


def get_command_logs(limit: Optional[int] = None) -> List[CommandLogEntry]:
    """Return the most recent command logs, optionally constrained by *limit*."""

    def _take_latest(items: Iterable[CommandLogEntry], count: Optional[int]) -> List[CommandLogEntry]:
        entries = list(items)
        if count is None or count >= len(entries):
            return entries
        return entries[-count:]

    with _COMMAND_LOGS_LOCK:
        snapshot = list(_COMMAND_LOGS)

    return _take_latest(snapshot, limit)


def build_inference_command(request: CentroidRequest, in_path: Path, out_path: Path) -> List[str]:
    """Builds the CLI command used to call the training script in inference mode."""

    cmd = [
        request.python_executable or sys.executable,
        request.train_script,
        "--mode",
        "infer",
        "--model-ckpt",
        request.model_ckpt,
        "--infer-geojson-in",
        str(in_path),
        "--infer-out",
        str(out_path),
        "--infer-knn",
        str(request.infer_params.knn),
        "--infer-e-thr",
        str(request.infer_params.e_thr),
        "--infer-il-thr",
        str(request.infer_params.il_thr),
        "--infer-sv1-thr",
        str(request.infer_params.sv1_thr),
        "--infer-slots",
        str(request.infer_params.slots),
        "--zone",
        request.zone_label,
    ]

    if request.config:
        cmd.extend(["--config", request.config])
    if request.device:
        cmd.extend(["--device", request.device])
    if request.services_target:
        cmd.extend(["--services-target", json.dumps(request.services_target)])
    if request.la_target is not None:
        cmd.extend(["--la-target", str(request.la_target)])
    if request.floors_avg is not None:
        cmd.extend(["--floors-avg", str(request.floors_avg)])

    return cmd


def _write_feature(path: Path, feature: Dict[str, Any]) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def run_centroid_inference(request: CentroidRequest) -> List[Dict[str, Any]]:
    """Executes the training script to obtain centroid predictions for a single block."""

    with tempfile.TemporaryDirectory(prefix="centroid_infer_") as tmp_dir:
        in_path = Path(tmp_dir) / "input.geojson"
        out_path = Path(tmp_dir) / "output.geojson"
        feature = dict(request.feature)
        feature_props = dict(feature.get("properties") or {})
        feature_props.setdefault(request.zone_attr, request.zone_label)
        feature["properties"] = feature_props
        _write_feature(in_path, feature)

        cmd = build_inference_command(request, in_path, out_path)
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        log_entry = CommandLogEntry(
            timestamp=datetime.now(timezone.utc),
            command=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
        _record_command_log(log_entry)
        if result.returncode != 0:
            raise RuntimeError(
                "Centroid inference command failed",
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "command": cmd,
                },
            )

        if not out_path.exists():
            raise RuntimeError("Inference output was not produced", {"command": cmd})

        try:
            output = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to parse inference output", {"path": str(out_path)}) from exc

        features = output.get("features", [])
        if not isinstance(features, list):
            raise RuntimeError("Malformed inference output: 'features' should be a list")

        return features


__all__ = [
    "CentroidRequest",
    "CentroidResult",
    "InferParams",
    "CommandLogEntry",
    "build_inference_command",
    "get_command_logs",
    "run_centroid_inference",
]