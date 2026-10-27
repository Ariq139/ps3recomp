import json
import sqlite3
from pathlib import Path

from .common import digest, json_text


SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts(
 id TEXT PRIMARY KEY, path TEXT NOT NULL, kind TEXT NOT NULL, metadata TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS inputs(
 artifact TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
 PRIMARY KEY(artifact,path));
CREATE TABLE IF NOT EXISTS functions(
 artifact TEXT NOT NULL, address INTEGER NOT NULL, name TEXT NOT NULL,
 source_hash TEXT NOT NULL DEFAULT '', metadata TEXT NOT NULL,
 PRIMARY KEY(artifact,address));
CREATE TABLE IF NOT EXISTS facts(
 id TEXT PRIMARY KEY, artifact TEXT NOT NULL, function INTEGER,
 kind TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL, evidence TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS edges(
 id TEXT PRIMARY KEY, artifact TEXT NOT NULL, caller INTEGER NOT NULL,
 target INTEGER, site INTEGER, kind TEXT NOT NULL, source TEXT NOT NULL,
 status TEXT NOT NULL, evidence TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS edges_caller ON edges(artifact,caller);
CREATE INDEX IF NOT EXISTS edges_target ON edges(artifact,target);
CREATE INDEX IF NOT EXISTS facts_function ON facts(artifact,function);
CREATE TABLE IF NOT EXISTS catalog(
 id TEXT PRIMARY KEY, artifact TEXT NOT NULL, kind TEXT NOT NULL,
 source TEXT NOT NULL, record TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(
 id TEXT PRIMARY KEY, artifact TEXT NOT NULL, function INTEGER NOT NULL,
 kind TEXT NOT NULL, proposed TEXT NOT NULL, status TEXT NOT NULL,
 evidence TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, summary TEXT NOT NULL);
"""


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA foreign_keys=ON")

    def close(self):
        self.db.close()

    def record(self, table, **values):
        keys = list(values)
        self.db.execute(f"INSERT OR REPLACE INTO {table} ({','.join(keys)}) "
                        f"VALUES ({','.join('?' for _ in keys)})", [values[k] for k in keys])

    def fact(self, artifact, function, kind, source, status, evidence):
        values = dict(artifact=artifact, function=function, kind=kind, source=source,
                      status=status, evidence=json_text(evidence))
        self.record("facts", id=digest(json_text(values).encode()), **values)

    def edge(self, artifact, caller, target, site, kind, source, status, evidence):
        values = dict(artifact=artifact, caller=caller, target=target, site=site,
                      kind=kind, source=source, status=status, evidence=json_text(evidence))
        self.record("edges", id=digest(json_text(values).encode()), **values)

    def candidate(self, artifact, function, kind, proposed, status, evidence):
        values = dict(artifact=artifact, function=function, kind=kind, proposed=proposed,
                      status=status, evidence=json_text(evidence))
        key = digest(json_text(values).encode())
        self.record("candidates", id=key, **values)
        return key

    def clear_source(self, artifact, source):
        for table in ("facts", "edges", "catalog"):
            self.db.execute(f"DELETE FROM {table} WHERE artifact=? AND source=?", (artifact, source))

    def rows(self, sql, params=()):
        return [dict(row) for row in self.db.execute(sql, params)]

    def summary(self, artifact):
        result = {"artifact_sha256": artifact}
        for table in ("functions", "facts", "edges", "catalog", "candidates"):
            result[table] = self.db.execute(f"SELECT COUNT(*) FROM {table} WHERE artifact=?", (artifact,)).fetchone()[0]
        result["edge_kinds"] = self.rows(
            "SELECT kind,status,COUNT(*) count FROM edges WHERE artifact=? GROUP BY kind,status", (artifact,))
        result["unique_call_sites"] = self.db.execute("SELECT COUNT(DISTINCT site) FROM edges WHERE artifact=? AND site IS NOT NULL", (artifact,)).fetchone()[0]
        return result
