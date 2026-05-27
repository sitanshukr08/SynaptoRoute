import sqlite3
import json
import os
from abc import ABC, abstractmethod
from typing import List

from synaptoroute.models import Route

class BaseStorage(ABC):
    @abstractmethod
    def save_route(self, route: Route):
        pass

    @abstractmethod
    def add_utterance(self, route_name: str, utterance: str):
        pass

    @abstractmethod
    def load_all_routes(self) -> List[Route]:
        pass

class SQLiteStorage(BaseStorage):
    def __init__(self, db_path: str):
        self.db_path = db_path
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.conn.execute('PRAGMA foreign_keys = ON')
        self._init_db()

    def _get_connection(self):
        return self.conn

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
                        FOREIGN KEY(route_name) REFERENCES routes(name)
                    )
                ''')
                conn.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to initialize database: {e}") from e

    def save_route(self, route: Route):
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
                    cursor.executemany('''
                        INSERT INTO utterances (route_name, utterance)
                        VALUES (?, ?)
                    ''', [(route.name, u) for u in route.utterances])
                    
                conn.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to save route: {e}") from e

    def add_utterance(self, route_name: str, utterance: str):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO utterances (route_name, utterance)
                    VALUES (?, ?)
                ''', (route_name, utterance))
                conn.commit()
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to add utterance: {e}") from e

    def load_all_routes(self) -> List[Route]:
        routes = []
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT name, threshold, metadata FROM routes')
                route_rows = cursor.fetchall()
                
                for row in route_rows:
                    name, threshold, metadata_str = row
                    metadata = json.loads(metadata_str) if metadata_str else None
                    
                    cursor.execute('SELECT utterance FROM utterances WHERE route_name = ?', (name,))
                    utterance_rows = cursor.fetchall()
                    utterances = [u[0] for u in utterance_rows]
                    
                    routes.append(Route(
                        name=name,
                        threshold=threshold,
                        metadata=metadata,
                        utterances=utterances
                    ))
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Failed to load routes: {e}") from e
        return routes
