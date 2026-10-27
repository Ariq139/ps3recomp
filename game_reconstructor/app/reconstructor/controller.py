import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .config import ROOT, sha256
from .engine import Engine
from .state import State, WorkspaceLock


class WorkerFailure(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Controller:
    def __init__(self, config, emit=None, stop=None):
        self.config = config
        self.emit = emit or (lambda event: print(json.dumps(event), flush=True))
        self.stop = stop or threading.Event()
        self.state = None
        self.engine = None
        self.profile = ""
        self.progress = {"status": "idle", "stage": "idle", "offline": True,
                         "reconstruction_complete": False, "game_build_run": False}
        self.draft_paths = set()

    def event(self, stage, **payload):
        event = {"stage": stage, **payload}
        if self.state:
            self.state.event(stage, payload)
            self.progress["tasks"] = self.state.counts(self.profile)
        self.progress.update(event)
        atomic_json(self.config.workspace / "progress.json", self.progress)
        self.emit(event)

    def worker(self, request):
        if sha256(request["binary"]) != request["expected_sha256"]:
            raise WorkerFailure("The local input changed after it was queued")
        jobs = self.config.workspace / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
        output = jobs / (key + (".jsonl" if request["action"] == "export" else ".json"))
        request = {**request, "workspace": str(self.config.workspace), "project": str(self.config.project),
                   "ghidra": str(self.config.ghidra), "language": self.config.language, "output": str(output)}
        request_path = jobs / (key + ".request.json")
        atomic_json(request_path, request)
        result_path = output.with_suffix(output.suffix + ".result.json")
        if output.is_file() and result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("output_sha256") == sha256(output):
                return output, cached
        log = jobs / (key + ".log")
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        with log.open("wb") as stream:
            process = subprocess.Popen([sys.executable, "-m", "reconstructor", "worker", str(request_path)],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, **kwargs)
            started, last_notice = time.monotonic(), time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic() - started > self.config.worker_timeout:
                        raise TimeoutError("The local Ghidra worker exceeded its configured time limit")
                    if time.monotonic() - last_notice > 30:
                        self.event("analyzing", worker_pid=process.pid, elapsed_seconds=int(time.monotonic() - started),
                                   pause_requested=self.stop.is_set())
                        last_notice = time.monotonic()
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=15)
        if process.returncode or not result_path.is_file():
            tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise WorkerFailure(f"Local analysis failed; log: {log}\n{tail}")
        return output, json.loads(result_path.read_text(encoding="utf-8"))

    def schedule(self, skip_references):
        self.state.begin_schedule(self.profile)
        manifest = self.engine.inventory()
        native_path = self.config.project / "out/EBOOT.functions.json"
        native = {int(row["start"], 16): int(row["end"], 16) for row in json.loads(native_path.read_text())} if native_path.is_file() else {}
        for row in manifest:
            named = not row["name"].startswith(("FUN_", "func_"))
            base = 10 if named and row["address"] in native else 20 if named else 30 if row["address"] in native else 40
            size = max(0, native.get(row["address"], row["address"] + 99999) - row["address"])
            self.state.add(self.engine.elf.sha256, row["address"], "release", self.engine.elf.path,
                           row["name"], base * 100000 + min(size, 99999), self.profile)
        if not skip_references:
            catalog_path = self.config.project / "symbols/EBOOT_PRIORITY_SYMBOL_LABELS.csv"
            with catalog_path.open(encoding="utf-8-sig", newline="") as stream:
                catalog = list(csv.DictReader(stream))
            for reference in self.config.references:
                if self.stop.is_set():
                    break
                identity = sha256(reference)
                selected = {}
                for row in catalog:
                    if row["source_artifact"] == reference.name:
                        selected[int(row["source_entry"], 16)] = {"name": row["source_name"], "size": int(row["source_size_bytes"] or 0)}
                if not selected:
                    self.event("reference_inventory", reference=str(reference))
                    _, inventory = self.worker({"action": "inventory", "binary": str(reference),
                                                 "expected_sha256": identity, "profile": self.profile})
                    selected = {row["address"]: row for row in inventory["functions"]}
                for ea, row in selected.items():
                    self.state.add(identity, ea, "reference", reference, row["name"], min(row["size"], 99999), self.profile)
        self.state.db.commit()
        self.event("queue_ready", queued=self.state.counts(self.profile))

    def analyze_batch(self, tasks):
        if not tasks or self.stop.is_set():
            return
        self.state.mark(tasks, "running")
        self.event("analyzing", role=tasks[0]["role"], functions=[f"0x{task['address']:08X}" for task in tasks])
        try:
            output, result = self.worker({"action": "export", "binary": tasks[0]["binary"],
                                          "expected_sha256": tasks[0]["artifact"], "profile": self.profile,
                                          "functions": [f"0x{task['address']:08X}" for task in tasks]})
            self.engine.import_export(output, tasks[0]["role"])
            records = {record["address"]: record for record in map(json.loads, output.read_text(encoding="utf-8").splitlines())
                       if record.get("kind") == "function"}
            for task in tasks:
                record = records.get(task["address"])
                if record is None or not record.get("coverage", {}).get("complete"):
                    self.state.mark([task], "blocked", "No complete decoded function body", str(output))
                else:
                    detail = "" if record.get("decompilation_status") == "completed" else "Pseudocode unavailable; native facts retained"
                    self.state.mark([task], "analyzed", detail, str(output))
            drafts = self.engine.export_drafts(output, self.config.workspace) if tasks[0]["role"] == "release" else []
            self.draft_paths.update(drafts)
            self.progress["readable_drafts"] = len(self.draft_paths)
            self.event("batch_analyzed", count=len(tasks), drafts_written=len(drafts), evidence=str(output))
        except (WorkerFailure, TimeoutError, ValueError, OSError) as error:
            self.engine.store.db.rollback()
            if len(tasks) > 1:
                self.state.mark(tasks, "pending", str(error))
                middle = len(tasks) // 2
                self.event("isolating_failure", count=len(tasks), reason=str(error))
                self.analyze_batch(tasks[:middle])
                self.analyze_batch(tasks[middle:])
            else:
                self.state.mark(tasks, "blocked", str(error))
                self.event("function_blocked", address=tasks[0]["address"], reason=str(error))

    def reference_artifacts(self):
        return {row["id"] for row in self.engine.store.rows("SELECT id FROM artifacts WHERE kind='ghidra_export'")
                if row["id"] != self.engine.elf.sha256}

    def refine(self, apply_verified):
        seen = set()
        for _ in range(4):
            result = self.engine.refine(self.reference_artifacts())
            self.event("connecting_and_rewriting", **result)
            if not apply_verified or not result["verified_decisions"]:
                break
            signature = result.get("verified_signature")
            if signature is not None and signature in seen:
                self.event("refinement_blocked", reason="Repeated decisions did not converge; implementations require review")
                break
            seen.add(signature)
            self.event("applying_verified_changes")
            self.event("changes_applied", **self.engine.apply_verified(self.config.workspace))

    def run(self, max_batches=None, max_functions=None, skip_references=False, analysis_only=False, retry_blocked=False):
        self.config.validate()
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        project_lock = self.config.project / "symbols/analysis/application.lock"
        with WorkspaceLock(self.config.workspace / "run.lock"), WorkspaceLock(project_lock):
            self.profile = self.config.profile()
            self.state = State(self.config.workspace / "state.sqlite")
            try:
                self.engine = Engine(self.config)
                self.event("indexing", status="running", profile=self.profile)
                self.event("indexed", **self.engine.index())
                self.draft_paths = {str(path) for path in (self.config.workspace / "readable_drafts").rglob("*.c")}
                self.progress.update(readable_drafts=len(self.draft_paths), validated_rewrites=self.engine.accepted_count())
                self.schedule(skip_references)
                if retry_blocked:
                    self.state.retry_blocked(self.profile)
                batches, attempted = 0, 0
                while not self.stop.is_set():
                    if max_batches is not None and batches >= max_batches:
                        break
                    remaining = None if max_functions is None else max_functions - attempted
                    if remaining is not None and remaining <= 0:
                        break
                    role = "release" if skip_references or batches % 4 else "reference"
                    size = self.config.batch_size if remaining is None else min(remaining, self.config.batch_size)
                    tasks = self.state.next_batch(self.profile, size, role)
                    if not tasks and not skip_references:
                        tasks = self.state.next_batch(self.profile, size)
                    if not tasks:
                        break
                    self.analyze_batch(tasks)
                    attempted += len(tasks)
                    batches += 1
                    if batches % 4 == 0:
                        self.refine(self.config.apply_verified and not analysis_only)
                self.refine(self.config.apply_verified and not analysis_only)
                self.event("writing_reports")
                report = self.engine.report()
                counts = self.state.counts(self.profile)
                status = "paused" if self.stop.is_set() else "bounded_run_finished" if counts.get("pending", 0) else "analysis_finished"
                self.progress.update(analysis=report, readable_drafts=len(list((self.config.workspace / "readable_drafts").rglob("*.c"))),
                                     validated_rewrites=self.engine.accepted_count(), attempted=attempted, batches=batches)
                self.event("finished", status=status, reconstruction_complete=False,
                           reason="Unsupported implementations remain queued for a validated rewrite rule")
                return dict(self.progress)
            except Exception as error:
                self.event("failed", status="failed", reason=str(error))
                raise
            finally:
                if self.engine:
                    self.engine.close()
                self.state.close()
                self.state = None
