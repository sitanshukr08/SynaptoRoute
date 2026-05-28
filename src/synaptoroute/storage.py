import sqlite3
import json
import os
import threading
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

class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.local = threading.local()
        self._init_db()

    def close(self):
        if hasattr(self.local, 'conn') and self.local.conn:
            self.local.conn.close()
            del self.local.conn

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_connection(self):
        if not hasattr(self.local, 'conn'):
            self.local.conn = sqlite3.connect(self.db_path, timeout=10.0)
            self.local.conn.execute('PRAGMA journal_mode=WAL;')
            self.local.conn.execute('PRAGMA foreign_keys = ON')
        return self.local.conn

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
            
        try:
            with self._get_connection() as conn:
                conn.execute('ALTER TABLE utterances ADD COLUMN embedding BLOB')
                conn.commit()
        except sqlite3.OperationalError:
            pass # Column already exists

    def save_route(self, route: Route, embeddings=None):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                metadata_str = json.dumps(route.metadata) if route.metadata is not None else None
                
                # Insert or replace route
                cursor.execute('''
                    INSERT OR REPLACE INTO routes (name, threshold, metadata)
                    VALUES (?, ?, ?)
                ''', (route.name, route.threshold, metadata_str))
                
                # Delete existing utterances for this route to avoid duplicates on replace
                cursor.execute('''
                    DELETE FROM utterances WHERE route_name = ?
                ''', (route.name,))
                
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
                cursor = conn.cursor()
                cursor.execute('SELECT name, threshold, metadata FROM routes')
                route_rows = cursor.fetchall()
                
                for row in route_rows:
                    name, threshold, metadata_str = row
                    metadata = json.loads(metadata_str) if metadata_str else None
                    
                    cursor.execute('SELECT utterance, embedding FROM utterances WHERE route_name = ?', (name,))
                    utterance_rows = cursor.fetchall()
                    utterances = []
                    embs = []
                    for u, e in utterance_rows:
                        utterances.append(u)
                        embs.append(e)
                    
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
