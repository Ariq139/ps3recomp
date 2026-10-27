import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reconstructor.config import Config, ROOT, sha256
from reconstructor.controller import Controller, WorkerFailure, atomic_json
from reconstructor.engine import Engine, bind_engine
from reconstructor.state import State, WorkspaceLock
from reconstructor.routing import source_folder


class FakeStore:
    db = SimpleNamespace(rollback=lambda: None)

    def rows(self, *args):
        return []


class FakeEngine:
    instances = []

    def __init__(self, config):
        self.config = config
        self.elf = SimpleNamespace(sha256=sha256(config.project / "game/EBOOT.ELF"), path=config.project / "game/EBOOT.ELF")
        self.store, self.applied = FakeStore(), 0
        self.instances.append(self)

    def inventory(self):
        return [{"address": ea, "name": "Known" + str(ea)} for ea in (0x100, 0x200)]

    def index(self):
        return {"functions": 2}

    def import_export(self, path, role):
        return {"functions_imported": 1}

    def export_drafts(self, path, workspace):
        return []

    def refine(self, refs):
        return {"verified_decisions": 0 if self.applied else 1}

    def apply_verified(self, workspace):
        self.applied += 1
        return {"decisions_applied": 1}

    def accepted_count(self):
        return 0

    def report(self):
        return {"functions": 2, "readability": {"lifted": 2}}

    def close(self):
        pass


class LocalController(Controller):
    failed_address = None

    def worker(self, request):
        addresses = [int(v, 16) for v in request["functions"]]
        if self.failed_address in addresses:
            raise WorkerFailure("fixture decoder failure")
        output = self.config.workspace / ("export_" + "_".join(request["functions"]) + ".jsonl")
        records = [{"kind": "metadata"}] + [{"kind": "function", "address": ea,
                   "coverage": {"complete": True}, "decompilation_status": "completed"} for ea in addresses]
        output.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return output, {"functions_exported": len(addresses)}


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="offline-reconstructor-tests-")
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        (self.project / "game").mkdir(parents=True)
        (self.project / "game/EBOOT.ELF").write_bytes(b"local fixture")
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.config = Config(self.project, self.root / "ghidra", self.workspace, references=[], batch_size=2)
        self.config.validate = lambda: {"offline": True}
        self.config.profile = lambda: "test-profile"
        FakeEngine.instances.clear()

    def tearDown(self):
        self.temporary.cleanup()

    def test_unknown_cloud_configuration_is_rejected(self):
        path = self.root / "profile.json"
        path.write_text(json.dumps({"project": "project", "ghidra": "ghidra", "provider": "cloud"}))
        with self.assertRaisesRegex(ValueError, "Unsupported settings"):
            Config.load(path)

    def test_unc_inputs_are_rejected_before_resolution(self):
        path = self.root / "profile.json"
        path.write_text(json.dumps({"project": "\\\\server\\game", "ghidra": "ghidra"}))
        with self.assertRaisesRegex(ValueError, "local disk"):
            Config.load(path)

    def test_queue_deduplicates_and_keeps_profiles_separate(self):
        state = State(self.workspace / "state.sqlite")
        try:
            for _ in range(2):
                state.add("image", 0x100, "release", "image.elf", "Known", 1, "first")
            state.add("image", 0x100, "release", "image.elf", "Known", 1, "second")
            state.db.commit()
            self.assertEqual(state.counts("first"), {"pending": 1})
            task = state.next_batch("first", 8)
            state.mark(task, "analyzed")
            self.assertEqual(state.next_batch("first", 8), [])
            self.assertEqual(state.counts("second"), {"pending": 1})
        finally:
            state.close()

    def test_interrupted_running_task_is_recovered(self):
        path = self.workspace / "state.sqlite"
        state = State(path)
        state.add("image", 0x100, "release", "image.elf", "Known", 1, "profile")
        state.mark(state.next_batch("profile", 1), "running")
        state.close()
        state = State(path)
        self.assertEqual(state.counts("profile"), {"pending": 1})
        state.close()

    def test_removed_reference_is_not_scheduled_again(self):
        state = State(self.workspace / "state.sqlite")
        try:
            state.add("reference", 0x100, "reference", "reference.gzf", "Source", 0, "profile")
            state.add("release", 0x200, "release", "game.elf", "Game", 1, "profile")
            state.begin_schedule("profile")
            state.add("release", 0x200, "release", "game.elf", "Game", 1, "profile")
            self.assertEqual(state.counts("profile"), {"pending": 1})
            self.assertEqual(state.next_batch("profile", 8)[0]["role"], "release")
        finally:
            state.close()

    def test_resume_refreshes_labels_without_repeating_analysis(self):
        state = State(self.workspace / "state.sqlite")
        try:
            state.add("image", 0x100, "release", "image.elf", "FUN_100", 1, "profile")
            state.mark(state.next_batch("profile", 1), "analyzed")
            state.add("image", 0x100, "release", "image.elf", "RecoveredName", 1, "profile")
            row = state.db.execute("SELECT name,status FROM tasks").fetchone()
            self.assertEqual((row["name"], row["status"]), ("RecoveredName", "analyzed"))
        finally:
            state.close()

    def test_workspace_lock_is_exclusive_and_reusable(self):
        path = self.workspace / "run.lock"
        with WorkspaceLock(path):
            with self.assertRaisesRegex(ValueError, "owns this workspace"):
                with WorkspaceLock(path):
                    pass
        with WorkspaceLock(path):
            pass

    def test_resume_does_not_repeat_finished_entries_or_claim_conversion(self):
        with patch("reconstructor.controller.Engine", FakeEngine):
            first = LocalController(self.config, emit=lambda _: None).run(max_functions=1, skip_references=True, analysis_only=True)
            second = LocalController(self.config, emit=lambda _: None).run(max_functions=1, skip_references=True, analysis_only=True)
        self.assertEqual(first["tasks"], {"analyzed": 1, "pending": 1})
        self.assertEqual(second["tasks"], {"analyzed": 2})
        self.assertEqual(second["attempted"], 1)
        self.assertFalse(second["reconstruction_complete"])
        self.assertEqual(second["analysis"]["readability"]["lifted"], 2)
        self.assertTrue(all(instance.applied == 0 for instance in FakeEngine.instances))

    def test_batch_failure_isolated_to_the_bad_function(self):
        with patch("reconstructor.controller.Engine", FakeEngine):
            controller = LocalController(self.config, emit=lambda _: None)
            controller.failed_address = 0x200
            result = controller.run(max_functions=2, skip_references=True, analysis_only=True)
        self.assertEqual(result["tasks"], {"analyzed": 1, "blocked": 1})
        self.assertFalse(result["reconstruction_complete"])

    def test_analysis_only_prevents_automatic_application(self):
        with patch("reconstructor.controller.Engine", FakeEngine):
            LocalController(self.config, emit=lambda _: None).run(max_functions=1, skip_references=True, analysis_only=True)
        self.assertEqual(FakeEngine.instances[-1].applied, 0)

    def test_verified_application_is_automatic_when_enabled(self):
        with patch("reconstructor.controller.Engine", FakeEngine):
            LocalController(self.config, emit=lambda _: None).run(max_functions=1, skip_references=True)
        self.assertEqual(FakeEngine.instances[-1].applied, 1)

    def test_stop_before_work_leaves_pending_entries(self):
        stop = threading.Event()
        stop.set()
        with patch("reconstructor.controller.Engine", FakeEngine):
            result = LocalController(self.config, emit=lambda _: None, stop=stop).run(skip_references=True, analysis_only=True)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["tasks"], {"pending": 2})

    def test_cached_worker_output_is_hash_checked(self):
        controller = Controller(self.config, emit=lambda _: None)
        binary = self.project / "game/EBOOT.ELF"
        request = {"action": "export", "binary": str(binary), "expected_sha256": sha256(binary), "functions": ["0x100"]}
        calls = []
        class Process:
            returncode, pid = 0, 123
            def poll(self):
                return 0
        def launch(command, **kwargs):
            calls.append(command)
            req = json.loads(Path(command[-1]).read_text())
            output = Path(req["output"])
            output.write_text("local export\n")
            atomic_json(output.with_suffix(output.suffix + ".result.json"), {"output_sha256": sha256(output)})
            return Process()
        with patch("reconstructor.controller.subprocess.Popen", side_effect=launch):
            output, _ = controller.worker(request)
            self.assertEqual(calls[0][:4], [sys.executable, "-m", "reconstructor", "worker"])
            controller.worker(request)
            self.assertEqual(len(calls), 1)
            output.write_text("changed")
            controller.worker(request)
            self.assertEqual(len(calls), 2)
            binary.write_bytes(b"different image")
            with self.assertRaisesRegex(WorkerFailure, "input changed"):
                controller.worker(request)

    def test_python_network_connections_are_blocked(self):
        code = "from reconstructor.offline import enforce_offline; enforce_offline(); import socket; socket.create_connection(('127.0.0.1',9))"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Offline reconstruction", result.stderr)

    def test_package_module_entry_point(self):
        result = subprocess.run([sys.executable, "-m", "reconstructor", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Offline Game Reconstructor", result.stdout)

    def test_source_library_names_route_without_guessing_callers(self):
        def function(name, aliases=""):
            return {"address": 0x100, "name": name, "metadata": json.dumps({"callable_names": aliases})}
        self.assertEqual(source_folder(function("criManaPlayer_Start"), {})[0], "criware")
        self.assertEqual(source_folder(function("getStatusSofdec"), {})[0], "criware")
        self.assertIsNone(source_folder(function("critical_damage"), {}))
        self.assertIsNone(source_folder(function("criManaWrapper", "SCRIPT_PLAY_MOVIE"), {}))
        self.assertEqual(source_folder(function("cellAudioInit"), {})[0], "cell")
        self.assertEqual(source_folder(function("FUN_00000100"), {"00000100": "battle/AI"})[0], "battle/AI")

    def test_readable_drafts_preserve_local_edits(self):
        bind_engine(self.project)
        engine = object.__new__(Engine)
        engine.config = self.config
        engine.inventory = lambda: [{"address": 0x100, "name": "KnownGetter", "metadata": json.dumps({"split_source": "split/battle/units/known.cpp"})}]
        output = self.workspace / "export.jsonl"
        record = {"kind": "function", "address": 0x100, "decompilation_status": "completed", "coverage": {"complete": True},
                  "pseudocode": "int FUN_00000100(void) { return 1; }", "body_sha256": "native"}
        output.write_text(json.dumps(record) + "\n")
        path = Path(engine.export_drafts(output, self.workspace)[0])
        self.assertIn("KnownGetter", path.read_text())
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertFalse(metadata["compilation_target"])
        path.write_text("hand edited implementation")
        incoming = Path(engine.export_drafts(output, self.workspace)[0])
        self.assertEqual(path.read_text(), "hand edited implementation")
        self.assertIn("incoming", incoming.parts)
        self.assertTrue(json.loads(incoming.with_suffix(".json").read_text())["local_edits_preserved"])
        incoming.write_text("edited incoming draft")
        next_incoming = Path(engine.export_drafts(output, self.workspace)[0])
        self.assertNotEqual(incoming, next_incoming)
        self.assertEqual(incoming.read_text(), "edited incoming draft")

    def test_complete_pseudocode_does_not_bypass_incomplete_native_coverage(self):
        bind_engine(self.project)
        engine = object.__new__(Engine)
        engine.config = self.config
        engine.inventory = lambda: [{"address": 0x100, "name": "KnownGetter", "metadata": json.dumps({"split_source": "split/unknown.cpp"})}]
        output = self.workspace / "partial.jsonl"
        output.write_text(json.dumps({"kind": "function", "address": 0x100,
                                      "decompilation_status": "completed", "coverage": {"complete": False}, "pseudocode": "return 1;"}) + "\n")
        self.assertEqual(engine.export_drafts(output, self.workspace), [])


if __name__ == "__main__":
    unittest.main()
