"""Versioned reference-sequence manifest and asset integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceCoacdCache:
    content_key: str
    path: Path
    sha256: str
    hull_count: int


@dataclass(frozen=True)
class ReferenceSequence:
    sequence_id: str
    robot_name: str
    source_dataset: str
    parquet_path: Path
    support_path: Path | None
    object_asset_paths: tuple[Path, ...]
    coacd_cache: ReferenceCoacdCache | None


@dataclass(frozen=True)
class ReferenceManifest:
    root: Path
    assets_root: Path
    sequences: tuple[ReferenceSequence, ...]

    def sequence(self, sequence_id: str) -> ReferenceSequence:
        matches = [
            entry for entry in self.sequences if entry.sequence_id == sequence_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous reference sequence: {sequence_id!r}")
        return matches[0]


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(
            f"reference asset is missing or escapes the repository: {relative}"
        )
    return candidate


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"reference asset hash mismatch for {path}: {actual} != {expected}"
        )


def load_reference_manifest(path: str | Path | None = None) -> ReferenceManifest:
    manifest_path = Path(path) if path is not None else default_manifest_path()
    root = manifest_path.resolve().parent.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"unsupported reference manifest schema: {payload.get('schema_version')}"
        )
    assets_root = (root / payload["assets_root"]).resolve()
    if not assets_root.is_relative_to(root) or not assets_root.is_dir():
        raise ValueError(f"reference assets root is not a directory: {assets_root}")
    sequences = []
    for entry in payload["sequences"]:
        parquet_path = _resolve(root, f"{payload['assets_root']}/{entry['parquet']}")
        _verify(parquet_path, entry["parquet_sha256"])
        object_paths = tuple(
            _resolve(root, f"{payload['assets_root']}/{relative}")
            for relative in entry["object_assets"]
        )
        support_path = (
            _resolve(root, f"{payload['assets_root']}/{entry['support']}")
            if entry.get("support")
            else None
        )
        for relative, expected in entry["asset_sha256"].items():
            _verify(_resolve(root, f"{payload['assets_root']}/{relative}"), expected)
        coacd = None
        if entry.get("coacd_cache") is not None:
            cache = entry["coacd_cache"]
            cache_path = _resolve(root, cache["path"])
            _verify(cache_path, cache["sha256"])
            coacd = ReferenceCoacdCache(
                content_key=str(cache["content_key"]),
                path=cache_path,
                sha256=str(cache["sha256"]),
                hull_count=int(cache["hull_count"]),
            )
        sequences.append(
            ReferenceSequence(
                sequence_id=str(entry["sequence_id"]),
                robot_name=str(entry["robot_name"]),
                source_dataset=str(entry["source_dataset"]),
                parquet_path=parquet_path,
                support_path=support_path,
                object_asset_paths=object_paths,
                coacd_cache=coacd,
            )
        )
    return ReferenceManifest(
        root=root, assets_root=assets_root, sequences=tuple(sequences)
    )


def reference_coacd_cache(
    sequence_id: str,
    content_key: str,
    *,
    manifest: ReferenceManifest | None = None,
) -> ReferenceCoacdCache | None:
    entry = (manifest or load_reference_manifest()).sequence(sequence_id)
    cache = entry.coacd_cache
    if cache is None:
        return None
    if cache.content_key != content_key:
        raise ValueError(
            f"reference CoACD key mismatch for {sequence_id}: {content_key} != {cache.content_key}"
        )
    return cache
