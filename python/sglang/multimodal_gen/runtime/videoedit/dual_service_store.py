# SPDX-License-Identifier: Apache-2.0
"""SQLite persistence for the single-consumer VideoEdit dual-service gateway."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = ("dispatching", "running", "cancelling")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


class DuplicateTaskError(ValueError):
    pass


class TaskNotFoundError(KeyError):
    pass


class DualServiceStore:
    """Small process-safe queue store.

    Connections are deliberately short lived. SQLite serializes writers and a
    partial unique index makes the global active-task invariant durable even if
    a second gateway worker is started by mistake.
    """

    _UPDATABLE_FIELDS = {
        "status",
        "started_at",
        "completed_at",
        "submitted_at",
        "backend_response",
        "error",
    }

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.path.abspath(os.fspath(path))
        parent = Path(self.path).parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    variant TEXT NOT NULL CHECK (variant IN ('normal', 'dmd')),
                    backend_url TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN (
                            'queued', 'dispatching', 'running', 'cancelling',
                            'completed', 'failed', 'cancelled'
                        )
                    ),
                    created_at REAL NOT NULL,
                    started_at REAL,
                    submitted_at REAL,
                    completed_at REAL,
                    backend_response TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS tasks_fifo
                ON tasks(status, created_at);

                CREATE UNIQUE INDEX IF NOT EXISTS tasks_one_active
                ON tasks ((1))
                WHERE status IN ('dispatching', 'running', 'cancelling');
                """)

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        task = dict(row)
        for field in ("request_json", "backend_response"):
            value = task.get(field)
            if value:
                try:
                    task[field] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    pass
        return task

    def enqueue(
        self,
        *,
        task_id: str,
        variant: str,
        backend_url: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_id, variant, backend_url, request_json, status, created_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?)
                    """,
                    (
                        task_id,
                        variant,
                        backend_url.rstrip("/"),
                        json.dumps(
                            request_payload, ensure_ascii=False, separators=(",", ":")
                        ),
                        now,
                    ),
                )
                connection.commit()
        except sqlite3.IntegrityError as error:
            if "tasks.task_id" in str(error) or "UNIQUE constraint failed" in str(
                error
            ):
                raise DuplicateTaskError(f"Task already exists: {task_id}") from error
            raise
        task = self.get(task_id)
        assert task is not None
        return task

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._decode_row(row)

    def get_active(self) -> dict[str, Any] | None:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM tasks
                WHERE status IN ({placeholders})
                ORDER BY created_at, rowid
                LIMIT 1
                """,
                ACTIVE_STATUSES,
            ).fetchone()
        return self._decode_row(row)

    def claim_next(self) -> dict[str, Any] | None:
        """Atomically claim one task only when no active task exists."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute("""
                SELECT task_id FROM tasks
                WHERE status IN ('dispatching', 'running', 'cancelling')
                LIMIT 1
                """).fetchone()
            if active is not None:
                connection.rollback()
                return None

            row = connection.execute("""
                SELECT task_id FROM tasks
                WHERE status = 'queued'
                ORDER BY created_at, rowid
                LIMIT 1
                """).fetchone()
            if row is None:
                connection.rollback()
                return None

            now = time.time()
            try:
                changed = connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'dispatching', started_at = ?, error = NULL
                    WHERE task_id = ? AND status = 'queued'
                    """,
                    (now, row["task_id"]),
                ).rowcount
                if changed != 1:
                    connection.rollback()
                    return None
                connection.commit()
            except sqlite3.IntegrityError:
                connection.rollback()
                return None

        return self.get(row["task_id"])

    def update_task(self, task_id: str, **updates: Any) -> dict[str, Any]:
        unknown = set(updates) - self._UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported task fields: {sorted(unknown)}")
        if not updates:
            task = self.get(task_id)
            if task is None:
                raise TaskNotFoundError(task_id)
            return task

        serialized = dict(updates)
        for field in ("backend_response",):
            if field in serialized and serialized[field] is not None:
                serialized[field] = json.dumps(
                    serialized[field], ensure_ascii=False, separators=(",", ":")
                )

        assignments = ", ".join(f"{field} = ?" for field in serialized)
        params = list(serialized.values()) + [task_id]
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                changed = connection.execute(
                    f"UPDATE tasks SET {assignments} WHERE task_id = ?", params
                ).rowcount
                if changed != 1:
                    connection.rollback()
                    raise TaskNotFoundError(task_id)
                connection.commit()
        except sqlite3.IntegrityError as error:
            raise RuntimeError(
                "Another active VideoEdit task already exists"
            ) from error

        task = self.get(task_id)
        assert task is not None
        return task

    def mark_terminal(
        self,
        task_id: str,
        status: str,
        *,
        backend_response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Not a terminal status: {status}")
        return self.update_task(
            task_id,
            status=status,
            completed_at=time.time(),
            backend_response=backend_response,
            error=error,
        )

    def cancel_queued(self, task_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE tasks
                SET status = 'cancelled', completed_at = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (time.time(), task_id),
            ).rowcount
            connection.commit()
        return changed == 1

    def list_tasks(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        query = "SELECT * FROM tasks"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at, rowid LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._decode_row(row) for row in rows]

    def queue_position(self, task_id: str) -> int | None:
        task = self.get(task_id)
        if task is None or task["status"] != "queued":
            return None
        with self._connect() as connection:
            value = connection.execute(
                """
                SELECT COUNT(*) FROM tasks
                WHERE status = 'queued'
                  AND (created_at < ? OR (created_at = ? AND rowid < (
                      SELECT rowid FROM tasks WHERE task_id = ?
                  )))
                """,
                (task["created_at"], task["created_at"], task_id),
            ).fetchone()[0]
        return int(value) + 1

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        counts = {
            status: 0 for status in (*ACTIVE_STATUSES, *TERMINAL_STATUSES, "queued")
        }
        counts.update({row["status"]: int(row["count"]) for row in rows})
        return counts
