import json
import os
import sqlite3
from pathlib import Path


class WorkspaceLock:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            self.stream = None
            raise ValueError("Another reconstruction run owns this workspace")
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


class State:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks(
              artifact TEXT NOT NULL, address INTEGER NOT NULL, role TEXT NOT NULL,
              binary TEXT NOT NULL, name TEXT NOT NULL, priority INTEGER NOT NULL,
              profile TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
              detail TEXT NOT NULL DEFAULT '', export TEXT NOT NULL DEFAULT '',
              enabled INTEGER NOT NULL DEFAULT 1,
              PRIMARY KEY(artifact,address,profile));
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
              stage TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        if "enabled" not in {row["name"] for row in self.db.execute("PRAGMA table_info(tasks)")}:
            self.db.execute("ALTER TABLE tasks ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
        self.db.execute("UPDATE tasks SET status='pending' WHERE status='running'")
        self.db.commit()

    def event(self, stage, payload):
        self.db.execute("INSERT INTO events(stage,payload) VALUES(?,?)", (stage, json.dumps(payload)))
        self.db.commit()

    def add(self, artifact, address, role, binary, name, priority, profile):
        self.db.execute("INSERT OR IGNORE INTO tasks(artifact,address,role,binary,name,priority,profile) VALUES(?,?,?,?,?,?,?)",
                        (artifact, address, role, str(binary), name, priority, profile))
        self.db.execute("UPDATE tasks SET enabled=1,name=?,priority=?,binary=? WHERE artifact=? AND address=? AND profile=?",
                        (name, priority, str(binary), artifact, address, profile))

    def begin_schedule(self, profile):
        self.db.execute("UPDATE tasks SET enabled=0 WHERE profile=?", (profile,))

    def next_batch(self, profile, size, role=None):
        condition, params = "profile=? AND status='pending' AND enabled=1", [profile]
        if role:
            condition += " AND role=?"
            params.append(role)
        first = self.db.execute(f"SELECT * FROM tasks WHERE {condition} ORDER BY priority,address LIMIT 1", params).fetchone()
        if not first:
            return []
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM tasks WHERE profile=? AND status='pending' AND enabled=1 AND artifact=? ORDER BY priority,address LIMIT ?",
            (profile, first["artifact"], size))]

    def mark(self, tasks, status, detail="", export=""):
        for task in tasks:
            self.db.execute("UPDATE tasks SET status=?,detail=?,export=? WHERE artifact=? AND address=? AND profile=?",
                            (status, detail, export, task["artifact"], task["address"], task["profile"]))
        self.db.commit()

    def counts(self, profile):
        return {row["status"]: row["count"] for row in self.db.execute(
            "SELECT status,COUNT(*) count FROM tasks WHERE profile=? AND enabled=1 GROUP BY status", (profile,))}

    def retry_blocked(self, profile):
        self.db.execute("UPDATE tasks SET status='pending',detail='' WHERE profile=? AND status='blocked'", (profile,))
        self.db.commit()

    def close(self):
        self.db.close()
