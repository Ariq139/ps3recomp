import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def local_path(value, base):
    if str(value).startswith(("\\\\", "//")):
        raise ValueError("Use a local disk path for offline operation")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass
class Config:
    project: Path
    ghidra: Path
    workspace: Path
    language: str = "PowerPC:BE:64:A2ALT-32addr"
    references: list[Path] = field(default_factory=list)
    batch_size: int = 8
    worker_timeout: int = 900
    apply_verified: bool = True

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        allowed = {"project", "ghidra", "workspace", "language", "references", "batch_size", "worker_timeout", "apply_verified"}
        if set(value) - allowed:
            raise ValueError("Unsupported settings: " + ", ".join(sorted(set(value) - allowed)))
        values = {k: v for k, v in value.items() if k not in ("project", "ghidra", "workspace", "references")}
        return cls(project=local_path(value["project"], path.parent),
                   ghidra=local_path(value["ghidra"], path.parent),
                   workspace=local_path(value.get("workspace", "workspaces/default"), path.parent),
                   references=[local_path(v, path.parent) for v in value.get("references", [])], **values)

    def save(self, path):
        value = {"project": str(self.project), "ghidra": str(self.ghidra), "workspace": str(self.workspace),
                 "language": self.language, "references": [str(v) for v in self.references],
                 "batch_size": self.batch_size, "worker_timeout": self.worker_timeout, "apply_verified": self.apply_verified}
        Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def validate(self):
        if any(str(path).startswith(("\\\\", "//")) for path in (self.project, self.ghidra, self.workspace, *self.references)):
            raise ValueError("Offline inputs and outputs must use local disk paths")
        if not isinstance(self.apply_verified, bool):
            raise ValueError("apply_verified must be true or false")
        if not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= 64:
            raise ValueError("Batch size must be between 1 and 64")
        if not isinstance(self.worker_timeout, int) or not 60 <= self.worker_timeout <= 86400:
            raise ValueError("Worker timeout must be between 60 and 86400 seconds")
        required = ["game/EBOOT.ELF", "symbols/EBOOT_GAME_FUNCTION_MODULES.csv",
                    "tools/split_lifted_ppu.py", "tools/ppu_readable_overrides.py"]
        for relative in required:
            if not (self.project / relative).is_file():
                raise ValueError(f"Target project is missing {relative}")
        if not (self.ghidra / "Ghidra/application.properties").is_file():
            raise ValueError("Choose the Ghidra installation directory")
        if importlib.util.find_spec("pyghidra") is None:
            raise ValueError("PyGhidra is required locally; this program does not download dependencies")
        for reference in self.references:
            if not reference.is_file():
                raise ValueError(f"Missing local reference: {reference}")
            if reference.is_relative_to(self.project / "src"):
                raise ValueError("A reference binary cannot be inside the generated source tree")
        if self.workspace == self.project or self.workspace.is_relative_to(self.project / "src"):
            raise ValueError("Use a separate workspace for analysis jobs")
        return {"offline": True, "cloud_services": False, "game_builds": False,
                "project": str(self.project), "ghidra": str(self.ghidra),
                "references": len(self.references)}

    def profile(self):
        parts = [self.language, sha256(self.ghidra / "Ghidra/application.properties")]
        parts.extend(sha256(ROOT / "reconstructor" / name) for name in ("engine.py", "worker.py"))
        parts.extend(sha256(path) for path in sorted((ROOT / "reconstructor/analysis_engine").rglob("*.py")))
        for name in ("split_lifted_ppu.py", "ppu_readable_overrides.py", "ppu_function_name_overrides.py", "ppu_function_path_overrides.py"):
            parts.append(sha256(self.project / "tools" / name))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()
