"""Generate a Markdown evidence table from verified claim manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.manifest_schema import validate_manifest_file  # noqa: E402


def generate(manifest_paths: list[Path]) -> str:
    rows = []
    for path in manifest_paths:
        errors = validate_manifest_file(path, repo_root=REPO_ROOT)
        if errors:
            raise ValueError(f"{path}: {errors}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "verified":
            raise ValueError(f"{path}: paper tables require verified evidence")
        rows.append(
            (
                manifest["benchmark"],
                manifest.get("claim", ""),
                manifest["git_commit"],
                manifest["archive"]["uri"],
            )
        )
    lines = [
        "| Benchmark | Claim | Commit | Archive |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| {benchmark} | {claim} | `{commit}` | {archive} |"
        for benchmark, claim, commit, archive in rows
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(generate(args.manifests), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
