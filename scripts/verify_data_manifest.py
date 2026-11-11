"""Check files against a SHA-256 manifest.

The manifest uses the shasum format, one '<64 hex digits>  <path>' line per file with paths
relative to the repository root, so it can also be checked without Python by running
`shasum -a 256 -c data/SHA256SUMS` from the root. When a listed directory is given, every file
in it except the manifest itself and dotfiles must be listed, and it may hold no symlink and no
directory that can't be read, so a new or renamed file cannot slip past the check.

Two manifests use this: data/SHA256SUMS pins the datasets, with data/ as the listed directory,
and tests/golden/SHA256SUMS pins the committed metrics, the reference out-of-fold run and the
saved model that the golden tests compare against.

Exit status: 0 when every listed file matches and nothing is unlisted, 1 otherwise, and 2 for a
usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path("data/SHA256SUMS")
DEFAULT_LISTED_DIR = Path("data")
CHUNK_BYTES = 1 << 20

# shasum writes a space for text mode and '*' for binary mode between digest and path. Both
# hash the same bytes on this platform, so both are accepted.
LINE_PATTERN = re.compile(r"^(?P<digest>[0-9a-f]{64}) [ *](?P<path>\S.*)$")


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks so large files stay out of memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_manifest(text: str) -> tuple[dict[str, str], list[str]]:
    """Return the listed path to digest mapping and a problem message for each bad line."""
    entries: dict[str, str] = {}
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = LINE_PATTERN.match(line)
        if match is None:
            problems.append(f"manifest line {number} is not '<sha256>  <path>': {line!r}")
            continue
        path = match["path"]
        if Path(path).is_absolute() or ".." in Path(path).parts:
            problems.append(f"manifest line {number} must name a path inside the root: {path}")
        elif path in entries:
            problems.append(f"manifest line {number} lists {path} a second time")
        else:
            entries[path] = match["digest"]
    return entries, problems


def _unlisted(manifest: Path, root: Path, listed_dir: Path, entries: dict[str, str]) -> list[str]:
    """Return a problem for each file under listed_dir that the manifest does not list.

    A symlink and a directory that can't be read are problems too, since either could hide a
    file from the walk, which follows no links.
    """
    base = root.resolve()
    directory = (root / listed_dir).resolve()
    if not directory.is_relative_to(base):
        return [f"{listed_dir}: listed directory must be inside the root"]
    if not directory.is_dir():
        state = "is not a directory" if directory.exists() else "does not exist"
        return [f"{listed_dir}: listed directory {state}"]
    found: list[tuple[str, str]] = []
    unreadable: list[OSError] = []
    manifest_file = manifest.resolve()
    for top, dirs, files in os.walk(directory, onerror=unreadable.append):
        for name in dirs + files:
            path = Path(top, name)
            relative = path.relative_to(base).as_posix()
            listable = name in files and not name.startswith(".") and path != manifest_file
            try:
                linked = path.is_symlink()
            except OSError as exc:
                found.append((relative, f"cannot be inspected ({type(exc).__name__}: {exc})"))
                continue
            if linked:
                found.append((relative, "is a symlink, so what it points to is not checked"))
            elif listable and relative not in entries:
                found.append((relative, "file is not listed in the manifest"))
    for error in unreadable:
        relative = Path(str(error.filename)).relative_to(base).as_posix()
        found.append((relative, f"directory cannot be read ({type(error).__name__}: {error})"))
    return [f"{relative}: {message}" for relative, message in sorted(found)]


def verify_manifest(manifest: Path, root: Path, listed_dir: Path | None) -> list[str]:
    """Return every reason the files under root do not match the manifest, empty when they do.

    listed_dir, relative to root, is the directory whose files must all be listed. Pass None
    when the manifest pins a chosen set of files spread across directories.
    """
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{manifest}: cannot read manifest ({exc})"]

    entries, problems = _parse_manifest(text)
    if not entries and not problems:
        problems.append(f"{manifest}: lists no files")

    for path, expected in entries.items():
        file = root / path
        try:
            if not file.is_file():
                state = "not a file" if file.exists() else "missing"
                problems.append(f"{path}: listed in the manifest but {state}")
                continue
            actual = sha256_of(file)
        except OSError as exc:
            problems.append(f"{path}: cannot be read ({type(exc).__name__}: {exc})")
            continue
        if actual != expected:
            problems.append(f"{path}: sha256 is {actual}, manifest expects {expected}")

    if listed_dir is not None:
        problems.extend(_unlisted(manifest, root, listed_dir, entries))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify a manifest and print each problem."""
    parser = argparse.ArgumentParser(description="Check files against a SHA-256 manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="root the manifest paths are relative to"
    )
    listed = parser.add_mutually_exclusive_group()
    listed.add_argument(
        "--listed-dir",
        type=Path,
        default=DEFAULT_LISTED_DIR,
        help="directory, relative to the root, whose files must all be listed",
    )
    listed.add_argument(
        "--no-listed-dir",
        action="store_true",
        help="check only the listed files, for a manifest that spans directories",
    )
    args = parser.parse_args(argv)

    manifest = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    listed_dir = None if args.no_listed_dir else args.listed_dir
    problems = verify_manifest(manifest, args.root, listed_dir)
    if problems:
        print(f"{len(problems)} problem(s) with {manifest}:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"Every file matches {manifest}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
