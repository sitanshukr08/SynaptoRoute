import sqlite3
import json
import os
import threading
import contextlib
import queue
from abc import ABC, abstractmethod
from typing import List

from synaptoroute.models import Route

class BaseStorage(ABC):
    @abstractmethod
    def save_route(self, route: Route, embeddings=None):
        pass

    @abstractmethod
    def add_utterance(self, route_name: str, utterance: str, embedding=None):
        pass

    @abstractmethod
    def update_threshold(self, route_name: str, threshold: float):
        pass

    @abstractmethod
    def load_all_routes(self) -> tuple[List[Route], dict]:
        pass

    @abstractmethod
    def delete_route(self, route_name: str):
        pass

    def delete_utterance(self, route_name: str, utterance: str):
        raise NotImplementedError

class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._memory_conn = None
        if self.db_path == ':memory:':
            self._memory_conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            self._configure_connection(self._memory_conn)
        else:
            dirname = os.path.dirname(self.db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
                
        self._pool = queue.Queue(maxsize=10)
        self._pool_sema = threading.Semaphore(10)
        
        self._init_db()

    def __del__(self):
        if hasattr(self, '_memory_conn') and self._memory_conn is not None:
            self._memory_conn.close()
        if hasattr(self, '_pool'):
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except queue.Empty:
                    break

    @contextlib.contextmanager
    def _get_connection(self):
        if self._memory_conn is not None:
            yield self._memory_conn
        else:
            self._pool_sema.acquire()
            try:
                try:
                    conn = self._pool.get_nowait()
                except queue.Empty:
                    conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
                    self._configure_connection(conn)
                try:
                    yield conn
                finally:
                    self._pool.put_nowait(conn)
            finally:
                self._pool_sema.release()

    def _configure_connection(self, conn: sqlite3.Connection):
        conn.isolation_level = "IMMEDIATE"
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA busy_timeout = 15000')

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS routes (
                        name TEXT PRIMARY KEY,
                        threshold REAL,
                        metadata TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS utterances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        route_name TEXT,
                        utterance TEXT,
                        FOREIGN KEY(route_name) REFERENCES routes(name) ON DELETE CASCADE,
                        UNIQUE(route_name, utterance)
                    )
                ''')
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to initialize database: {e}") from e
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA table_info(utterances)')
            columns = [info[1] for info in cursor.fetchall()]
            if 'embedding' not in columns:
                conn.execute('ALTER TABLE utterances ADD COLUMN embedding BLOB')
            conn.commit()

    def save_route(self, route: Route, embeddings=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                metadata_str = json.dumps(route.metadata) if route.metadata is not None else None
                
                # Delete existing utterances for this route to avoid duplicates on replace
                cursor.execute('''
                    DELETE FROM utterances WHERE route_name = ?
                ''', (route.name,))
                
                # Insert or replace route
                cursor.execute('''
                    INSERT OR REPLACE INTO routes (name, threshold, metadata)
                    VALUES (?, ?, ?)
                ''', (route.name, route.threshold, metadata_str))
                
                # Insert utterances
                if route.utterances:
                    if embeddings is not None:
                        cursor.executemany('''
                            INSERT OR IGNORE INTO utterances (route_name, utterance, embedding)
                            VALUES (?, ?, ?)
                        ''', [(route.name, u, e.tobytes()) for u, e in zip(route.utterances, embeddings)])
                    else:
                        cursor.executemany('''
                            INSERT OR IGNORE INTO utterances (route_name, utterance)
                            VALUES (?, ?)
                        ''', [(route.name, u) for u in route.utterances])
                    
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to save route: {e}") from e

    def update_threshold(self, route_name: str, threshold: float):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE routes SET threshold = ? WHERE name = ?
                ''', (threshold, route_name))
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to update threshold: {e}") from e

    def add_utterance(self, route_name: str, utterance: str, embedding=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO utterances (route_name, utterance, embedding)
                    VALUES (?, ?, ?)
                ''', (route_name, utterance, embedding.tobytes() if embedding is not None else None))
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to add utterance: {e}") from e

    def load_all_routes(self) -> tuple[List[Route], dict]:
        routes = []
        embeddings_map = {}
        try:
            with self._get_connection() as conn:
                original_isolation = conn.isolation_level
                try:
                    conn.isolation_level = None
                    cursor = conn.cursor()
                    cursor.execute('BEGIN IMMEDIATE')
                    
                    cursor.execute('SELECT name, threshold, metadata FROM routes')
                    route_rows = cursor.fetchall()
                    
                    cursor.execute('SELECT route_name, utterance, embedding FROM utterances')
                    utterance_rows = cursor.fetchall()
                    
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.isolation_level = original_isolation
                
                utt_dict = {}
                emb_dict = {}
                for route_name, utt, emb in utterance_rows:
                    if route_name not in utt_dict:
                        utt_dict[route_name] = []
                        emb_dict[route_name] = []
                    utt_dict[route_name].append(utt)
                    emb_dict[route_name].append(emb)
                
                for row in route_rows:
                    name, threshold, metadata_str = row
                    try:
                        metadata = json.loads(metadata_str) if metadata_str else None
                    except json.JSONDecodeError:
                        metadata = None
                    
                    utterances = utt_dict.get(name, [])
                    embs = emb_dict.get(name, [])
                    
                    routes.append(Route(
                        name=name,
                        threshold=threshold,
                        metadata=metadata,
                        utterances=utterances
                    ))
                    embeddings_map[name] = embs
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to load routes: {e}") from e
        return routes, embeddings_map

    def delete_route(self, route_name: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM routes WHERE name = ?', (route_name,))
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to delete route: {e}") from e

    def delete_utterance(self, route_name: str, utterance: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'DELETE FROM utterances WHERE route_name = ? AND utterance = ?',
                    (route_name, utterance),
                )
                conn.commit()
        except (sqlite3.OperationalError, sqlite3.IntegrityError, sqlite3.DatabaseError) as e:
            raise RuntimeError(f"Failed to delete utterance: {e}") from e
