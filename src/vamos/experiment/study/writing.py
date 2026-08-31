"""Atomic file primitives shared by study creation and execution."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .serialization import stored_document_bytes


def write_document_atomic(path: Path, document: Mapping[str, Any]) -> bytes:
    """Atomically replace one mutable checkpoint with canonical bytes."""
    payload = stored_document_bytes(document)
    write_bytes_atomic(path, payload)
    return payload


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish one file atomically after flushing its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning("Could not remove owned temporary file %s", temporary, exc_info=True)


def fsync_directory(path: Path) -> None:
    """Flush a directory entry where the platform exposes that primitive."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["fsync_directory", "write_bytes_atomic", "write_document_atomic"]
