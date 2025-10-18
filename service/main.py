from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from scripts.centroid_runner import (
    CentroidRequest,
    CommandLogEntry,
    get_command_logs,
    run_centroid_inference,
)

app = FastAPI(title="Centroid Inference Service")

_TASK_HISTORY_LIMIT = 200
_TASKS: Dict[str, "_TaskRecord"] = {}
_TASK_ORDER = deque()
_TASKS_LOCK = Lock()
_MISSING = object()


@dataclass
class _TaskRecord:
    """Represents execution metadata for a centroid generation request."""

    task_id: str
    request_id: Optional[str]
    status: str
    zone_label: str
    created_at: datetime
    updated_at: datetime
    features_count: Optional[int] = None
    detail: Optional[str] = None

    def serialize(self) -> dict:
        return {
            "task_id": self.task_id,
            "request_id": self.request_id,
            "status": self.status,
            "zone_label": self.zone_label,
            "features_count": self.features_count,
            "detail": self.detail,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _register_task(request: CentroidRequest) -> _TaskRecord:
    record = _TaskRecord(
        task_id=str(uuid4()),
        request_id=request.request_id,
        status="running",
        zone_label=request.zone_label,
        created_at=_now(),
        updated_at=_now(),
    )
    with _TASKS_LOCK:
        _TASKS[record.task_id] = record
        _TASK_ORDER.append(record.task_id)
        while len(_TASK_ORDER) > _TASK_HISTORY_LIMIT:
            oldest = _TASK_ORDER.popleft()
            _TASKS.pop(oldest, None)
    return record


def _update_task(
    task_id: str,
    *,
    status: str,
    detail: object = _MISSING,
    features_count: object = _MISSING,
) -> None:
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return
        task.status = status
        task.updated_at = _now()
        if detail is not _MISSING:
            task.detail = detail if detail is None or isinstance(detail, str) else str(detail)
        if features_count is not _MISSING:
            task.features_count = features_count if isinstance(features_count, int) else None


def _get_tasks_snapshot() -> List[dict]:
    with _TASKS_LOCK:
        return [
            _TASKS[task_id].serialize()
            for task_id in list(_TASK_ORDER)
            if task_id in _TASKS
        ]


def _format_error_detail(exc: RuntimeError) -> str:
    if len(exc.args) > 1:
        payload = exc.args[1]
        if isinstance(payload, (dict, list, tuple)):
            try:
                return json.dumps(payload)
            except TypeError:
                return repr(payload)
        return str(payload)
    return str(exc)


@app.post("/centroids")
def generate_centroids(request: CentroidRequest):
    """Run centroid inference for a single block via the training script."""

    task = _register_task(request)
    try:
        features = run_centroid_inference(request)
    except RuntimeError as exc:  # pragma: no cover - service level error propagation
        _update_task(task.task_id, status="failed", detail=_format_error_detail(exc))
        detail = exc.args[1] if len(exc.args) > 1 else {"message": str(exc)}
        raise HTTPException(status_code=500, detail=detail) from exc
    except Exception as exc:  # pragma: no cover - unexpected service failure
        _update_task(task.task_id, status="failed", detail=str(exc))
        raise

    _update_task(task.task_id, status="succeeded", features_count=len(features), detail=None)
    return {"task_id": task.task_id, "features": features}


@app.get("/logs")
def fetch_logs(limit: Optional[int] = Query(None, ge=1)):
    """Expose the captured stdout/stderr emitted by inference subprocesses."""

    logs = get_command_logs(limit=limit)
    return {"logs": [_serialize_log(entry) for entry in logs]}


@app.get("/tasks")
def list_tasks():
    """Return the recent centroid generation tasks and their status."""

    return {"tasks": _get_tasks_snapshot()}


def _serialize_log(entry: CommandLogEntry) -> dict:
    """Convert a command log entry into a JSON-serializable payload."""

    payload = entry.dict()
    payload["timestamp"] = entry.timestamp.isoformat()
    return payload