"""Restore the complete SCI reproducibility archive. Requires Python 3."""
from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile
import hashlib

root = Path(__file__).resolve().parent
target = root / "sci-repro"
if target.exists():
    raise SystemExit("sci-repro already exists. Restore into a fresh directory.")
digest = hashlib.sha256()
with TemporaryFile() as buf:
    for i in range(1, 8 + 1):
        part = root / f"archive-{i:02d}.part"
        if not part.is_file():
            raise SystemExit(f"Missing {part.name}")
        with part.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                buf.write(block)
    if digest.hexdigest() != "57d6073f14fb6e4d8312b1b2c7d12e09c9a5d7d923c1062f4ce9d2f3c0f55159":
        raise SystemExit("Archive checksum mismatch. Download all parts again.")
    buf.seek(0)
    with ZipFile(buf) as archive:
        archive.extractall(root)
print("Restored sci-repro/. See sci-repro/README.md for reproduction commands.")
