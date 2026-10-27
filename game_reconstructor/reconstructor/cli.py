import argparse
import json
import signal
import threading
from pathlib import Path

from .config import Config, ROOT
from .offline import enforce_offline


def gui_main():
    return main(["gui"])


def main(argv=None):
    enforce_offline()
    parser = argparse.ArgumentParser(description="Offline Game Reconstructor: local analysis, readable drafts, and verified implementation changes")
    commands = parser.add_subparsers(dest="command", required=True)
    default = ROOT / "reconstructor/profiles/p5_npeb02436.json"
    for name in ("run", "gui", "doctor", "status"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, default=default)
        if name == "run":
            command.add_argument("--max-batches", type=int)
            command.add_argument("--max-functions", type=int)
            command.add_argument("--analysis-only", action="store_true")
            command.add_argument("--skip-references", action="store_true")
            command.add_argument("--retry-blocked", action="store_true")
        if name == "gui":
            command.add_argument("--check", action="store_true", help="Construct the interface without starting analysis")
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import run_gui
            return run_gui(args.profile, args.check)
        if args.command == "worker":
            from .worker import execute
            result = execute(args.request)
            print(json.dumps({"local_worker_finished": True, "functions": result.get("functions_exported", len(result.get("functions", []))),
                              "output_sha256": result["output_sha256"]}, indent=2))
            return 0
        config = Config.load(args.profile)
        if args.command == "doctor":
            print(json.dumps(config.validate(), indent=2))
        elif args.command == "status":
            path = config.workspace / "progress.json"
            print(path.read_text(encoding="utf-8") if path.is_file() else '{"status":"not_started"}')
        else:
            if args.max_batches is not None and args.max_batches < 1 or args.max_functions is not None and args.max_functions < 1:
                raise ValueError("Run limits must be positive")
            from .controller import Controller
            stop = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            Controller(config, stop=stop).run(args.max_batches, args.max_functions, args.skip_references, args.analysis_only, args.retry_blocked)
        return 0
    except Exception as error:
        parser.exit(2, f"Reconstruction stopped: {error}\n")
