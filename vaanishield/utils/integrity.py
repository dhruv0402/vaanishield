"""Local artifact integrity checks. A digest is not proof of a trusted publisher."""
import hashlib
import re
from pathlib import Path
from typing import Optional


def verify_sha256_sidecar(filepath: str, payload: Optional[bytes] = None) -> None:
    """Require a valid matching SHA-256 sidecar before deserializing an artifact.

    Accept both ``model.npz.sha256`` and the legacy ``model.sha256`` name.
    Callers must obtain the model and digest from a trusted source. Replacing both
    defeats this corruption check, and a checksum cannot make pickle safe.
    When a caller already read the artifact, pass those exact bytes as payload
    so the checked data is the data subsequently deserialized.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Target model file not found: {path}")
    sha_path = path.with_suffix(path.suffix + ".sha256")
    if not sha_path.is_file():
        sha_path = path.with_suffix(".sha256")
    if not sha_path.is_file():
        raise FileNotFoundError(f"SHA-256 sidecar missing for {path}")
    fields = sha_path.read_text(encoding="utf-8").split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ValueError(f"Invalid SHA-256 sidecar: {sha_path}")
    digest = hashlib.sha256()
    if payload is not None:
        digest.update(payload)
    else:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    if digest.hexdigest() != fields[0].lower():
        raise ValueError(f"Cryptographic integrity mismatch for {path}")
