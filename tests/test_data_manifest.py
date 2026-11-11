"""Tests for scripts/verify_data_manifest.py against the committed datasets and small copies."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.verify_data_manifest import main, sha256_of, verify_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "SHA256SUMS"
DATA = Path("data")


def _listed_paths(manifest: Path) -> list[str]:
    """Return the paths a manifest lists, in file order."""
    return [line.split(maxsplit=1)[1] for line in manifest.read_text().splitlines() if line]


def _small_project(root: Path, files: dict[str, bytes]) -> Path:
    """Write files under root plus a manifest of their real hashes, returning the manifest."""
    lines = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        lines.append(f"{sha256_of(path)}  {relative}")
    manifest = root / "data" / "SHA256SUMS"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


@pytest.fixture
def copied_datasets(tmp_path: Path) -> Path:
    """A copy of the committed datasets and manifest, safe to modify."""
    for relative in [*_listed_paths(MANIFEST), "data/SHA256SUMS"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    return tmp_path


def test_datasets_match_manifest() -> None:
    """The manifest lists exactly the two committed datasets and both hash to what it records."""
    assert verify_manifest(MANIFEST, REPO_ROOT, DATA) == []
    assert len(_listed_paths(MANIFEST)) == 2


def test_one_byte_change_is_detected(copied_datasets: Path) -> None:
    """Changing one byte of the last listed dataset is reported against that file."""
    last = _listed_paths(MANIFEST)[-1]
    path = copied_datasets / last
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0x01
    path.write_bytes(bytes(content))

    problems = verify_manifest(copied_datasets / "data" / "SHA256SUMS", copied_datasets, DATA)
    assert len(problems) == 1
    assert problems[0].startswith(f"{last}: sha256 is ")


@settings(max_examples=60)
@given(
    content=st.binary(min_size=1, max_size=256),
    position=st.integers(min_value=0),
    flip=st.integers(min_value=1, max_value=255),
)
def test_any_single_byte_flip_is_detected(
    tmp_path_factory: pytest.TempPathFactory, content: bytes, position: int, flip: int
) -> None:
    """Flipping any bits of any single byte of a listed file is always reported."""
    root = tmp_path_factory.mktemp("flip")
    manifest = _small_project(root, {"data/raw/a.csv": b"x,y\n", "data/b.csv": content})
    changed = bytearray(content)
    changed[position % len(changed)] ^= flip
    (root / "data/b.csv").write_bytes(bytes(changed))

    problems = verify_manifest(manifest, root, DATA)
    assert [p.split(":")[0] for p in problems] == ["data/b.csv"]


def test_missing_file_is_detected(tmp_path: Path) -> None:
    """A listed file that no longer exists, or is now a directory, is a problem named as such."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n", "data/b.csv": b"2\n"})
    (tmp_path / "data/a.csv").unlink()
    (tmp_path / "data/b.csv").unlink()
    (tmp_path / "data/b.csv").mkdir()
    assert verify_manifest(manifest, tmp_path, DATA) == [
        "data/a.csv: listed in the manifest but missing",
        "data/b.csv: listed in the manifest but not a file",
    ]


def test_unlisted_dataset_is_detected(tmp_path: Path) -> None:
    """Any file under data/ that the manifest does not list is a problem, whatever its type and
    also inside a hidden directory. Only dotfiles themselves are exempt."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    (tmp_path / "data/new").mkdir()
    for name in ["extra.csv", "chain.parquet", "UPPER.CSV"]:
        (tmp_path / "data/new" / name).write_bytes(b"3\n")
    (tmp_path / "data/.DS_Store").write_bytes(b"finder")
    (tmp_path / "data/.hidden").mkdir()
    (tmp_path / "data/.hidden/extra.csv").write_bytes(b"4\n")
    assert verify_manifest(manifest, tmp_path, DATA) == [
        "data/.hidden/extra.csv: file is not listed in the manifest",
        "data/new/UPPER.CSV: file is not listed in the manifest",
        "data/new/chain.parquet: file is not listed in the manifest",
        "data/new/extra.csv: file is not listed in the manifest",
    ]
    assert verify_manifest(manifest, tmp_path, None) == []


@pytest.mark.parametrize(
    ("listed_dir", "message"),
    [
        pytest.param(Path("absent"), "absent: listed directory does not exist", id="missing"),
        pytest.param(Path(".."), "..: listed directory must be inside the root", id="outside"),
        pytest.param(Path("a.txt"), "a.txt: listed directory is not a directory", id="a_file"),
    ],
)
def test_unusable_listed_directory_is_a_problem(
    tmp_path: Path, listed_dir: Path, message: str
) -> None:
    """A listed directory that is absent, outside the root or a file fails instead of passing."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    (tmp_path / "a.txt").write_text("not a directory\n")
    assert verify_manifest(manifest, tmp_path, listed_dir) == [message]


def test_binary_mode_line_is_accepted(tmp_path: Path) -> None:
    """shasum's binary-mode separator ' *' is accepted alongside the text-mode two spaces."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    manifest.write_text(manifest.read_text().replace("  data/a.csv", " *data/a.csv"))
    assert verify_manifest(manifest, tmp_path, DATA) == []


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("not a manifest line", id="free_text"),
        pytest.param("abc123  data/a.csv", id="short_digest"),
        pytest.param(
            "06CAD809709C8A1AFB207C0653E447AAA65349E1292925806ABAEA527DD269DF  data/a.csv",
            id="uppercase_digest",
        ),
        pytest.param(f"{'0' * 64} data/a.csv", id="single_space"),
        pytest.param(f"{'0' * 64}  /etc/passwd", id="absolute_path"),
        pytest.param(f"{'0' * 64}  data/../outside.csv", id="parent_path"),
    ],
)
def test_malformed_manifest_line_is_detected(tmp_path: Path, line: str) -> None:
    """A line that is not a valid in-root sha256sum entry is reported with its line number."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    manifest.write_text(manifest.read_text() + line + "\n")
    problems = verify_manifest(manifest, tmp_path, DATA)
    assert any(p.startswith("manifest line 2 ") for p in problems), problems


def test_duplicate_manifest_path_is_detected(tmp_path: Path) -> None:
    """Listing the same path twice is a problem even when both digests are right."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    manifest.write_text(manifest.read_text() * 2)
    assert verify_manifest(manifest, tmp_path, DATA) == [
        "manifest line 2 lists data/a.csv a second time"
    ]


def test_empty_or_unreadable_manifest_is_a_problem(tmp_path: Path) -> None:
    """An empty manifest verifies nothing and an absent one cannot be read, both fail."""
    manifest = _small_project(tmp_path, {})
    assert verify_manifest(manifest, tmp_path, DATA) == [f"{manifest}: lists no files"]
    problems = verify_manifest(tmp_path / "absent", tmp_path, DATA)
    assert len(problems) == 1
    assert "cannot read manifest" in problems[0]


def test_manifest_hashes_agree_with_shasum() -> None:
    """The system hasher, independent of hashlib, accepts the committed manifest."""
    shasum = shutil.which("shasum") or shutil.which("sha256sum")
    if shasum is None:
        pytest.fail("neither shasum nor sha256sum is on PATH")
    command = [shasum, "-a", "256", "-c"] if shasum.endswith("shasum") else [shasum, "-c"]
    result = subprocess.run(
        [*command, "data/SHA256SUMS"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count(": OK") == len(_listed_paths(MANIFEST))


def test_unreadable_listed_file_is_a_problem(tmp_path: Path) -> None:
    """A listed file that can't be read is reported, and the other files are still checked."""
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n", "data/b.csv": b"2\n"})
    (tmp_path / "data/b.csv").write_bytes(b"changed\n")
    locked = tmp_path / "data/a.csv"
    locked.chmod(0o000)
    try:
        problems = verify_manifest(manifest, tmp_path, DATA)
    finally:
        locked.chmod(0o644)
    assert len(problems) == 2
    assert problems[0].startswith("data/a.csv: cannot be read (PermissionError")
    assert problems[1].startswith("data/b.csv: sha256 is ")


@pytest.mark.parametrize("kind", ["directory", "file", "dangling"])
def test_symlink_in_listed_dir_is_a_problem(tmp_path: Path, kind: str) -> None:
    """A symlink under the listed directory is reported, whatever it points to.

    The walk follows no links, so files behind a linked directory would otherwise go unseen.
    """
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "extra.csv").write_bytes(b"3\n")
    targets = {"directory": outside, "file": outside / "extra.csv", "dangling": outside / "gone"}
    (tmp_path / "data/linked").symlink_to(targets[kind])
    assert verify_manifest(manifest, tmp_path, DATA) == [
        "data/linked: is a symlink, so what it points to is not checked"
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param(0o000, "data/locked: directory cannot be read", id="no_access"),
        pytest.param(0o644, "data/locked/extra.csv: cannot be inspected", id="no_search"),
    ],
)
def test_unreadable_directory_in_listed_dir_is_a_problem(
    tmp_path: Path, mode: int, expected: str
) -> None:
    """A directory under the listed one that can't be read or searched is reported, not raised.

    Without read permission the walk can't list it. With read but no search permission it lists
    the names, but none of them can be inspected.
    """
    manifest = _small_project(tmp_path, {"data/a.csv": b"1\n"})
    locked = tmp_path / "data/locked"
    locked.mkdir()
    (locked / "extra.csv").write_bytes(b"3\n")
    locked.chmod(mode)
    try:
        problems = verify_manifest(manifest, tmp_path, DATA)
    finally:
        locked.chmod(0o755)
    assert len(problems) == 1
    assert problems[0].startswith(f"{expected} (PermissionError")


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exits 0 on a match, 1 on any problem and 2 on a usage error, checks the manifest
    that --manifest names, and runs as a script."""
    _small_project(tmp_path, {"data/a.csv": b"1\n"})
    assert main(["--root", str(tmp_path)]) == 0
    (tmp_path / "data/a.csv").write_bytes(b"2\n")
    assert main(["--root", str(tmp_path)]) == 1
    assert "data/a.csv: sha256 is" in capsys.readouterr().out

    # Only the manifest named here matches the changed file, so passing shows it was the one used.
    elsewhere = tmp_path / "elsewhere.sums"
    elsewhere.write_text(f"{sha256_of(tmp_path / 'data/a.csv')}  data/a.csv\n")
    assert main(["--root", str(tmp_path), "--manifest", str(elsewhere), "--no-listed-dir"]) == 0
    assert f"Every file matches {elsewhere}." in capsys.readouterr().out
    golden = ["--manifest", "tests/golden/SHA256SUMS", "--no-listed-dir"]
    assert main(["--root", str(REPO_ROOT), *golden]) == 0
    assert f"Every file matches {REPO_ROOT / 'tests/golden/SHA256SUMS'}." in capsys.readouterr().out
    with pytest.raises(SystemExit) as usage_error:
        main(["--listed-dir", "data", "--no-listed-dir"])
    assert usage_error.value.code == 2

    script = [sys.executable, "scripts/verify_data_manifest.py"]
    result = subprocess.run(script, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    # As a script on the changed file, the process itself must exit 1, which make verify_data needs.
    failing = [*script, "--root", str(tmp_path)]
    result = subprocess.run(failing, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
