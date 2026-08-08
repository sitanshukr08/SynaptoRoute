"""Build a content-inventoried source and evidence archive for external deposit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import sha256_file, validate_manifest  # noqa: E402


ARCHIVE_SCHEMA_VERSION = 1
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
LABEL_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


@dataclass(frozen=True)
class ArchiveInput:
    label: str
    path: Path


@dataclass(frozen=True)
class ArchiveMember:
    archive_path: str
    role: str
    source_path: Path | None = None
    content: bytes | None = None

    def __post_init__(self):
        if (self.source_path is None) == (self.content is None):
            raise ValueError("archive members require exactly one content source")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def parse_archive_input(value: str) -> ArchiveInput:
    label, separator, path_text = value.partition("=")
    if not separator or not LABEL_PATTERN.fullmatch(label) or not path_text:
        raise argparse.ArgumentTypeError("inputs must use LABEL=PATH with a safe non-empty label")
    path = Path(path_text).resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"archive input does not exist: {path}")
    return ArchiveInput(label=label, path=path)


def _tracked_files(repo_root: Path) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        stderr=subprocess.DEVNULL,
    )
    files = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = repo_root / raw_path.decode("utf-8")
        if not path.is_file():
            raise RuntimeError(f"tracked source file is missing: {path}")
        if path.is_symlink():
            raise RuntimeError(f"source symlinks are not allowed in the archive: {path}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(repo_root).as_posix())


def _input_files(archive_input: ArchiveInput) -> list[tuple[str, Path]]:
    if archive_input.path.is_symlink():
        raise RuntimeError(f"evidence symlinks are not allowed: {archive_input.path}")
    if archive_input.path.is_file():
        return [(archive_input.path.name, archive_input.path)]
    files = []
    for path in archive_input.path.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"evidence symlinks are not allowed: {path}")
        if path.is_file():
            files.append((path.relative_to(archive_input.path).as_posix(), path))
    if not files:
        raise RuntimeError(f"evidence input is empty: {archive_input.path}")
    return sorted(files)


def _resolve_manifest_path(repo_root: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _validate_evidence_manifest(
    path: Path,
    *,
    repo_root: Path,
    commit: str,
    included_paths: set[Path],
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest, repo_root=repo_root)
    if errors:
        raise RuntimeError(f"invalid evidence manifest {path}: {errors}")
    if manifest.get("git_commit") != commit:
        raise RuntimeError(
            f"evidence manifest commit mismatch: {path} has {manifest.get('git_commit')}, "
            f"candidate is {commit}"
        )
    if manifest.get("working_tree_dirty") is not False:
        raise RuntimeError(f"evidence manifest was not produced from a clean tree: {path}")
    if manifest.get("exit_status") != 0:
        raise RuntimeError(f"evidence manifest records a failed run: {path}")

    evidence = manifest.get("evidence", {})
    raw_path = _resolve_manifest_path(repo_root, evidence.get("raw_output_path"))
    raw_sha256 = evidence.get("raw_output_sha256")
    if raw_path is None or not raw_path.is_file() or not raw_sha256:
        raise RuntimeError(f"manifest lacks hashed raw output: {path}")
    if sha256_file(raw_path) != raw_sha256:
        raise RuntimeError(f"manifest raw-output hash mismatch: {path}")
    if raw_path.resolve() not in included_paths:
        raise RuntimeError(f"manifest raw output is not included in evidence input: {path}")

    dependency_lock = manifest.get("dependency_lock")
    if isinstance(dependency_lock, dict):
        lock_path = _resolve_manifest_path(repo_root, dependency_lock.get("path"))
        if lock_path is None or not lock_path.is_file():
            raise RuntimeError(f"manifest dependency lock is missing: {path}")
        if sha256_file(lock_path) != dependency_lock.get("sha256"):
            raise RuntimeError(f"manifest dependency-lock hash mismatch: {path}")
    return manifest


def _sha256_path_exact(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_record(member: ArchiveMember) -> dict[str, Any]:
    if member.source_path is not None:
        size_bytes = member.source_path.stat().st_size
        digest = _sha256_path_exact(member.source_path)
    else:
        assert member.content is not None
        size_bytes = len(member.content)
        digest = _sha256_bytes(member.content)
    return {
        "path": member.archive_path,
        "role": member.role,
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def _write_zip(path: Path, members: Iterable[ArchiveMember]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for member in sorted(members, key=lambda value: value.archive_path):
                info = zipfile.ZipInfo(member.archive_path, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                with archive.open(info, mode="w", force_zip64=True) as destination:
                    if member.source_path is not None:
                        with member.source_path.open("rb") as source:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                    else:
                        assert member.content is not None
                        destination.write(member.content)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_archive(
    *,
    repo_root: Path,
    inputs: list[ArchiveInput],
    output_path: Path,
    require_clean: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_path = output_path.resolve()
    if output_path.suffix.lower() != ".zip":
        raise ValueError("archive output must use the .zip extension")
    if not inputs:
        raise ValueError("at least one evidence input is required")
    if len({item.label for item in inputs}) != len(inputs):
        raise ValueError("archive input labels must be unique")
    for item in inputs:
        if item.path.is_file():
            if output_path == item.path:
                raise ValueError("archive output cannot replace an evidence input")
            continue
        try:
            output_path.relative_to(item.path)
        except ValueError:
            continue
        raise ValueError("archive output cannot be inside an evidence input")

    commit = _git(repo_root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("archive requires a full Git commit SHA")
    dirty = bool(_git(repo_root, "status", "--porcelain"))
    if require_clean and dirty:
        raise RuntimeError("archive requires a clean working tree")

    payload_members: list[ArchiveMember] = []
    inventory: list[dict[str, Any]] = []
    for source_path in _tracked_files(repo_root):
        archive_path = f"source/{source_path.relative_to(repo_root).as_posix()}"
        member = ArchiveMember(
            archive_path=archive_path,
            role="source",
            source_path=source_path,
        )
        payload_members.append(member)
        inventory.append(_member_record(member))

    manifest_records = []
    for item in sorted(inputs, key=lambda value: value.label):
        files = _input_files(item)
        included_paths = {path.resolve() for _, path in files}
        manifests = [
            (relative, path)
            for relative, path in files
            if Path(relative).name == "manifest.json"
        ]
        if not manifests:
            raise RuntimeError(f"evidence input contains no manifest.json: {item.path}")
        for relative_path, manifest_path in manifests:
            manifest = _validate_evidence_manifest(
                manifest_path,
                repo_root=repo_root,
                commit=commit,
                included_paths=included_paths,
            )
            manifest_records.append(
                {
                    "input": item.label,
                    "run_id": manifest.get("run_id"),
                    "benchmark": manifest["benchmark"],
                    "status": manifest["status"],
                    "path": f"evidence/{item.label}/{relative_path}",
                }
            )
        for relative_path, evidence_path in files:
            archive_path = f"evidence/{item.label}/{relative_path}"
            member = ArchiveMember(
                archive_path=archive_path,
                role=f"evidence:{item.label}",
                source_path=evidence_path,
            )
            payload_members.append(member)
            inventory.append(_member_record(member))

    with (repo_root / "pyproject.toml").open("rb") as handle:
        package_version = tomllib.load(handle)["project"]["version"]
    lock_path = repo_root / "paper" / "requirements-linux-py311.lock"
    metadata = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "project": "SynaptoRoute",
        "package_version": package_version,
        "git_commit": commit,
        "git_commit_timestamp": _git(repo_root, "show", "-s", "--format=%cI", commit),
        "working_tree_dirty": dirty,
        "dependency_lock": {
            "path": lock_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(lock_path),
        },
        "inputs": [{"label": item.label} for item in sorted(inputs, key=lambda value: value.label)],
        "manifests": sorted(
            manifest_records,
            key=lambda item: (item["input"], item["benchmark"], str(item["run_id"])),
        ),
        "notes": [
            "Archive membership is listed in ARCHIVE_INVENTORY.json.",
            "Verified claims require a separate independently reproduced promotion manifest.",
        ],
    }
    metadata_content = _json_bytes(metadata)
    metadata_member = ArchiveMember(
        archive_path="ARCHIVE_METADATA.json",
        role="archive_metadata",
        content=metadata_content,
    )
    inventory.append(_member_record(metadata_member))
    inventory_content = _json_bytes(
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "entry_count": len(inventory),
            "entries": sorted(inventory, key=lambda item: item["path"]),
        }
    )
    payload_members.extend(
        [
            metadata_member,
            ArchiveMember(
                archive_path="ARCHIVE_INVENTORY.json",
                role="archive_inventory",
                content=inventory_content,
            ),
        ]
    )
    _write_zip(output_path, payload_members)
    archive_sha256 = sha256_file(output_path)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".sha256")
    sidecar_path.write_text(f"{archive_sha256}  {output_path.name}\n", encoding="ascii")
    return {
        "archive_path": output_path.as_posix(),
        "archive_sha256": archive_sha256,
        "sha256_sidecar_path": sidecar_path.as_posix(),
        "entry_count": len(payload_members),
        "manifest_count": len(manifest_records),
        "git_commit": commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=parse_archive_input,
        required=True,
        dest="inputs",
        help="Evidence input as LABEL=PATH; repeat for multiple result roots.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_archive(
        repo_root=REPO_ROOT,
        inputs=args.inputs,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
