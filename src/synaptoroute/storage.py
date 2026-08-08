from __future__ import annotations

import contextlib
import json
import os
import queue
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Literal

from synaptoroute.exceptions import StorageVersionConflictError
from synaptoroute.models import Route


SynchronousMode = Literal["FULL", "NORMAL"]
LATEST_SCHEMA_VERSION = 3


class BaseStorage(ABC):
    @abstractmethod
    def save_route(
        self,
        route: Route,
        embeddings=None,
        expected_version: int | None = None,
    ):
        pass

    @abstractmethod
    def add_utterance(
        self,
        route_name: str,
        utterance: str,
        embedding=None,
        version: int | None = None,
    ):
        pass

    @abstractmethod
    def update_threshold(
        self,
        route_name: str,
        threshold: float,
        version: int | None = None,
    ):
        pass

    @abstractmethod
    def load_all_routes(self) -> tuple[list[Route], dict[str, list[bytes | None]]]:
        pass

    @abstractmethod
    def delete_route(self, route_name: str, expected_version: int | None = None):
        pass

    def delete_utterance(self, route_name: str, utterance: str):
        raise NotImplementedError


class SQLiteStorage(BaseStorage):
    """Versioned SQLite persistence with explicit read and write transactions."""

    def __init__(
        self,
        db_path: str,
        synchronous: SynchronousMode = "FULL",
        pool_size: int = 10,
    ):
        normalized_mode = synchronous.upper()
        if normalized_mode not in {"FULL", "NORMAL"}:
            raise ValueError("synchronous must be 'FULL' or 'NORMAL'")
        if pool_size < 1:
            raise ValueError("pool_size must be positive")

        self.db_path = db_path
        self.synchronous: SynchronousMode = normalized_mode  # type: ignore[assignment]
        self._closed = False
        self._memory_conn: sqlite3.Connection | None = None
        self._memory_lock = threading.RLock()
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=pool_size)
        self._pool_sema = threading.BoundedSemaphore(pool_size)

        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(
                self.db_path,
                timeout=15.0,
                check_same_thread=False,
                isolation_level=None,
            )
            self._configure_connection(self._memory_conn)
        else:
            dirname = os.path.dirname(os.path.abspath(self.db_path))
            os.makedirs(dirname, exist_ok=True)

        self._init_db()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
        while not self._pool.empty():
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @contextlib.contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        if self._closed:
            raise RuntimeError("SQLiteStorage is closed")
        if self._memory_conn is not None:
            with self._memory_lock:
                yield self._memory_conn
            return

        self._pool_sema.acquire()
        conn: sqlite3.Connection | None = None
        try:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=15.0,
                    check_same_thread=False,
                    isolation_level=None,
                )
                self._configure_connection(conn)
            yield conn
        finally:
            if conn is not None:
                if self._closed:
                    conn.close()
                else:
                    self._pool.put_nowait(conn)
            self._pool_sema.release()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute(f"PRAGMA synchronous={self.synchronous}")

    @contextlib.contextmanager
    def _transaction(
        self,
        conn: sqlite3.Connection,
        *,
        write: bool,
    ) -> Iterator[sqlite3.Cursor]:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield cursor
            conn.commit()
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def _columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
        cursor.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cursor.fetchall()}

    def _init_db(self) -> None:
        migrations = (
            self._migration_1_baseline,
            self._migration_2_versioned_embeddings,
            self._migration_3_deterministic_order,
        )
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute("SELECT version FROM schema_migrations")
            applied = {int(row[0]) for row in cursor.fetchall()}
            for version, migration in enumerate(migrations, start=1):
                if version in applied:
                    continue
                migration(cursor)
                cursor.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    (version,),
                )

    def _migration_1_baseline(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS routes (
                name TEXT PRIMARY KEY,
                threshold REAL NOT NULL,
                metadata TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_name TEXT NOT NULL,
                utterance TEXT NOT NULL,
                FOREIGN KEY(route_name) REFERENCES routes(name) ON DELETE CASCADE,
                UNIQUE(route_name, utterance)
            )
            """
        )

    def _migration_2_versioned_embeddings(self, cursor: sqlite3.Cursor) -> None:
        if "version" not in self._columns(cursor, "routes"):
            cursor.execute(
                "ALTER TABLE routes ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        if "embedding" not in self._columns(cursor, "utterances"):
            cursor.execute("ALTER TABLE utterances ADD COLUMN embedding BLOB")

    def _migration_3_deterministic_order(self, cursor: sqlite3.Cursor) -> None:
        if "position" not in self._columns(cursor, "utterances"):
            cursor.execute("ALTER TABLE utterances ADD COLUMN position INTEGER")
        cursor.execute(
            """
            UPDATE utterances
            SET position = (
                SELECT COUNT(*) - 1
                FROM utterances AS earlier
                WHERE earlier.route_name = utterances.route_name
                  AND earlier.id <= utterances.id
            )
            WHERE position IS NULL
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_utterances_route_position "
            "ON utterances(route_name, position)"
        )

    @staticmethod
    def _current_version(cursor: sqlite3.Cursor, route_name: str) -> int | None:
        cursor.execute("SELECT version FROM routes WHERE name = ?", (route_name,))
        row = cursor.fetchone()
        return int(row[0]) if row is not None else None

    def _advance_version(
        self,
        cursor: sqlite3.Cursor,
        route_name: str,
        version: int | None,
        *,
        threshold: float | None = None,
    ) -> None:
        if version is None:
            return
        expected = version - 1
        if threshold is None:
            cursor.execute(
                "UPDATE routes SET version = ? WHERE name = ? AND version = ?",
                (version, route_name, expected),
            )
        else:
            cursor.execute(
                "UPDATE routes SET threshold = ?, version = ? "
                "WHERE name = ? AND version = ?",
                (threshold, version, route_name, expected),
            )
        if cursor.rowcount != 1:
            raise StorageVersionConflictError(
                route_name,
                expected,
                self._current_version(cursor, route_name),
            )

    def save_route(
        self,
        route: Route,
        embeddings=None,
        expected_version: int | None = None,
    ) -> None:
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            metadata = json.dumps(route.metadata) if route.metadata is not None else None
            if expected_version is None:
                cursor.execute(
                    """
                    INSERT INTO routes(name, threshold, version, metadata)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        threshold=excluded.threshold,
                        version=excluded.version,
                        metadata=excluded.metadata
                    """,
                    (route.name, route.threshold, route.version, metadata),
                )
            elif expected_version == 0:
                try:
                    cursor.execute(
                        "INSERT INTO routes(name, threshold, version, metadata) "
                        "VALUES (?, ?, ?, ?)",
                        (route.name, route.threshold, route.version, metadata),
                    )
                except sqlite3.IntegrityError as error:
                    raise StorageVersionConflictError(
                        route.name,
                        expected_version,
                        self._current_version(cursor, route.name),
                    ) from error
            else:
                cursor.execute(
                    "UPDATE routes SET threshold = ?, version = ?, metadata = ? "
                    "WHERE name = ? AND version = ?",
                    (
                        route.threshold,
                        route.version,
                        metadata,
                        route.name,
                        expected_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageVersionConflictError(
                        route.name,
                        expected_version,
                        self._current_version(cursor, route.name),
                    )
            cursor.execute("DELETE FROM utterances WHERE route_name = ?", (route.name,))
            embedding_rows = list(embeddings) if embeddings is not None else []
            rows = []
            for position, utterance in enumerate(route.utterances):
                embedding = embedding_rows[position] if position < len(embedding_rows) else None
                rows.append(
                    (
                        route.name,
                        utterance,
                        embedding.tobytes() if embedding is not None else None,
                        position,
                    )
                )
            cursor.executemany(
                "INSERT INTO utterances(route_name, utterance, embedding, position) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def add_utterance(
        self,
        route_name: str,
        utterance: str,
        embedding=None,
        version: int | None = None,
    ) -> None:
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            self._advance_version(cursor, route_name, version)
            cursor.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM utterances WHERE route_name = ?",
                (route_name,),
            )
            position = int(cursor.fetchone()[0])
            cursor.execute(
                "INSERT INTO utterances(route_name, utterance, embedding, position) "
                "VALUES (?, ?, ?, ?)",
                (
                    route_name,
                    utterance,
                    embedding.tobytes() if embedding is not None else None,
                    position,
                ),
            )

    def update_threshold(
        self,
        route_name: str,
        threshold: float,
        version: int | None = None,
    ) -> None:
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            if version is None:
                cursor.execute(
                    "UPDATE routes SET threshold = ? WHERE name = ?",
                    (threshold, route_name),
                )
            else:
                self._advance_version(
                    cursor,
                    route_name,
                    version,
                    threshold=threshold,
                )

    def load_route(self, route_name: str) -> tuple[Route | None, list[bytes | None]]:
        routes, embeddings = self._load_routes(route_name)
        if not routes:
            return None, []
        return routes[0], embeddings[routes[0].name]

    def load_all_routes(self) -> tuple[list[Route], dict[str, list[bytes | None]]]:
        return self._load_routes(None)

    def _load_routes(
        self,
        route_name: str | None,
    ) -> tuple[list[Route], dict[str, list[bytes | None]]]:
        with self._get_connection() as conn, self._transaction(conn, write=False) as cursor:
            if route_name is None:
                cursor.execute(
                    "SELECT name, threshold, version, metadata FROM routes ORDER BY name"
                )
                route_rows = cursor.fetchall()
                cursor.execute(
                    "SELECT route_name, utterance, embedding FROM utterances "
                    "ORDER BY route_name, position, id"
                )
            else:
                cursor.execute(
                    "SELECT name, threshold, version, metadata FROM routes WHERE name = ?",
                    (route_name,),
                )
                route_rows = cursor.fetchall()
                cursor.execute(
                    "SELECT route_name, utterance, embedding FROM utterances "
                    "WHERE route_name = ? ORDER BY position, id",
                    (route_name,),
                )
            utterance_rows = cursor.fetchall()

        utterances: dict[str, list[str]] = {}
        embeddings: dict[str, list[bytes | None]] = {}
        for stored_name, utterance, embedding in utterance_rows:
            utterances.setdefault(stored_name, []).append(utterance)
            embeddings.setdefault(stored_name, []).append(embedding)

        routes: list[Route] = []
        for stored_name, threshold, version, metadata_json in route_rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else None
            except json.JSONDecodeError:
                metadata = None
            routes.append(
                Route(
                    name=stored_name,
                    threshold=threshold,
                    version=version,
                    metadata=metadata,
                    utterances=utterances.get(stored_name, []),
                )
            )
            embeddings.setdefault(stored_name, [])
        return routes, embeddings

    def delete_route(self, route_name: str, expected_version: int | None = None) -> None:
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            if expected_version is None:
                cursor.execute("DELETE FROM routes WHERE name = ?", (route_name,))
            else:
                cursor.execute(
                    "DELETE FROM routes WHERE name = ? AND version = ?",
                    (route_name, expected_version),
                )
                if cursor.rowcount != 1:
                    raise StorageVersionConflictError(
                        route_name,
                        expected_version,
                        self._current_version(cursor, route_name),
                    )

    def delete_utterance(self, route_name: str, utterance: str) -> None:
        with self._get_connection() as conn, self._transaction(conn, write=True) as cursor:
            cursor.execute(
                "DELETE FROM utterances WHERE route_name = ? AND utterance = ?",
                (route_name, utterance),
            )
