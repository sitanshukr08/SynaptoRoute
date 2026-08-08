"""Independently verify a SynaptoRoute evidence archive and its inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


def _stream_sha256(source: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return _stream_sha256(source)[0]


def _safe_archive_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\\" not in value


def _sidecar_digest(archive_path: Path) -> str | None:
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if not sidecar.is_file():
        return None
    parts = sidecar.read_text(encoding="ascii").strip().split()
    if len(parts) < 2 or parts[-1] != archive_path.name:
        raise RuntimeError(f"invalid archive SHA-256 sidecar: {sidecar}")
    digest = parts[0]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError(f"invalid archive SHA-256 sidecar digest: {sidecar}")
    return digest


def verify_archive(
    archive_path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict:
    archive_path = archive_path.resolve()
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"archive does not exist or is empty: {archive_path}")
    actual_archive_sha256 = _file_sha256(archive_path)
    sidecar_sha256 = _sidecar_digest(archive_path)
    expected = expected_sha256 or sidecar_sha256
    if expected is not None and actual_archive_sha256 != expected:
        raise RuntimeError(
            f"archive SHA-256 mismatch: expected {expected}, found {actual_archive_sha256}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate member names")
        if "ARCHIVE_INVENTORY.json" not in names or "ARCHIVE_METADATA.json" not in names:
            raise RuntimeError("archive is missing inventory or metadata")
        inventory = json.loads(archive.read("ARCHIVE_INVENTORY.json"))
        metadata = json.loads(archive.read("ARCHIVE_METADATA.json"))
        if inventory.get("schema_version") != 1 or not isinstance(inventory.get("entries"), list):
            raise RuntimeError("archive inventory schema is invalid")
        if metadata.get("schema_version") != 1:
            raise RuntimeError("archive metadata schema is invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("git_commit", ""))):
            raise RuntimeError("archive metadata does not contain a full Git commit")

        entries = inventory["entries"]
        if inventory.get("entry_count") != len(entries):
            raise RuntimeError("archive inventory entry count is inconsistent")
        records_by_path = {}
        for record in entries:
            if not isinstance(record, dict) or not _safe_archive_path(str(record.get("path", ""))):
                raise RuntimeError("archive inventory contains an unsafe or invalid path")
            path = str(record["path"])
            if path in records_by_path:
                raise RuntimeError(f"archive inventory contains a duplicate path: {path}")
            if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
                raise RuntimeError(f"archive inventory has an invalid size: {path}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))):
                raise RuntimeError(f"archive inventory has an invalid hash: {path}")
            records_by_path[path] = record

        expected_names = set(records_by_path) | {"ARCHIVE_INVENTORY.json"}
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise RuntimeError(f"archive membership differs from inventory; missing={missing}, extra={extra}")

        for path, record in sorted(records_by_path.items()):
            with archive.open(path) as source:
                digest, size = _stream_sha256(source)
            if size != record["size_bytes"]:
                raise RuntimeError(f"archive member size mismatch: {path}")
            if digest != record["sha256"]:
                raise RuntimeError(f"archive member SHA-256 mismatch: {path}")

    return {
        "archive_path": archive_path.as_posix(),
        "archive_sha256": actual_archive_sha256,
        "sidecar_verified": sidecar_sha256 is not None,
        "inventory_entry_count": len(records_by_path),
        "git_commit": metadata["git_commit"],
        "package_version": metadata["package_version"],
        "manifest_count": len(metadata.get("manifests", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    report = verify_archive(args.archive, expected_sha256=args.sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
