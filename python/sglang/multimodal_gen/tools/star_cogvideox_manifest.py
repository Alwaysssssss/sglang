"""Manifest helpers for STAR CogVideoX-SR asset conversion."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hash for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    sha256: str | None = None


@dataclass
class SourceAssetsManifest:
    transformer_checkpoint: FileRecord
    vae_checkpoint: FileRecord | None = None
    text_encoder_dir: str | None = None
    tokenizer_dir: str | None = None
    text_encoder_files: list[FileRecord] = field(default_factory=list)
    tokenizer_files: list[FileRecord] = field(default_factory=list)
    config_path: FileRecord | None = None


@dataclass
class ComponentExportRecord:
    component_name: str
    output_dir: str
    output_files: list[str] = field(default_factory=list)
    tensor_count: int | None = None
    parameter_count: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class KeyMappingSummary:
    component_name: str
    source_key_count: int
    exported_key_count: int
    stripped_prefixes: list[str] = field(default_factory=list)
    dropped_key_prefixes: list[str] = field(default_factory=list)
    dropped_key_count: int = 0


@dataclass
class ConversionReport:
    source_format: str
    output_dir: str
    pipeline_class_name: str
    components: list[ComponentExportRecord] = field(default_factory=list)
    key_mapping: list[KeyMappingSummary] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def describe_file(
    path: str | os.PathLike[str], *, include_hash: bool = True
) -> FileRecord:
    path = str(path)
    stat = os.stat(path)
    return FileRecord(
        path=path,
        size_bytes=stat.st_size,
        sha256=sha256_file(path) if include_hash else None,
    )


def describe_directory_files(
    directory: str | os.PathLike[str], *, include_hash: bool = True
) -> list[FileRecord]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    records: list[FileRecord] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            records.append(describe_file(entry, include_hash=include_hash))
    return records


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_dataclass_json(path: str | os.PathLike[str], payload: Any) -> None:
    write_json(path, asdict(payload))
