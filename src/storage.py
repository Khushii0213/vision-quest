"""Local JSON persistence for best score, totals and run history (no database needed)."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MAX_RUNS = 50


class StorageError(RuntimeError):
    """The progress file could not be written."""


def _default() -> dict:
    return {"version": 1, "best_score": 0, "best_streak": 0, "rounds_played": 0, "correct_total": 0,
            "wrong_total": 0, "answered_total": 0, "response_time_total": 0.0, "runs": []}


class Storage:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        if not self.path.exists():
            return _default()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("runs", []), list):
                raise ValueError("unexpected structure")
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            # Keep the unreadable file for inspection instead of silently deleting it.
            try:
                os.replace(self.path, self.path.with_suffix(".corrupt"))
            except OSError:
                pass
            return _default()
        return {**_default(), **data}

    def _write(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            raise StorageError(f"Could not save progress to {self.path}: {exc}") from exc

    def record_run(self, summary: dict, now: datetime | None = None) -> dict:
        """Append a finished run. Returns {'new_best': bool, 'previous_best': int}."""
        data = self.load()
        previous = data["best_score"]
        stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        data["runs"] = (data["runs"] + [{**summary, "timestamp": stamp}])[-MAX_RUNS:]
        data["best_score"] = max(previous, summary["score"])
        data["best_streak"] = max(data["best_streak"], summary["best_streak"])
        data["rounds_played"] += summary["rounds_played"]
        data["correct_total"] += summary["correct"]
        data["wrong_total"] += summary["wrong"]
        data["answered_total"] += summary["answered"]
        data["response_time_total"] = round(data["response_time_total"] + summary["response_time_total"], 3)
        self._write(data)
        return {"new_best": summary["score"] > previous, "previous_best": previous}

    def reset(self) -> None:
        self._write(_default())

    @staticmethod
    def average_response(data: dict) -> float | None:
        return round(data["response_time_total"] / data["answered_total"], 2) if data["answered_total"] else None
