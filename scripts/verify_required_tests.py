"""Check that every required test id passed, using the JUnit XML reports of each toolchain.

A required id names one test case in one language:

    py::tests/test_split.py::test_folds_are_ordered
    py::tests/test_split.py::TestPurge::test_boundary[63]
    java::com.example.pricing.BlackScholesTest::putCallParity
    ts::src/blotter.test.ts::blotter > sorts by vrp

Each id must appear in exactly one report for its language and must have passed. An id that
is missing, skipped, failed, errored, flaky or found in more than one report is a problem, as
is an unreadable report, so a required test cannot pass vacuously.

Exit status: 0 when every required id passed, 1 when there are problems, 2 when the file
listing the required ids is unusable.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree.ElementTree import Element, ParseError

from defusedxml import DefusedXmlException, ElementTree

Status = Literal["passed", "flaky", "skipped", "failed", "errored"]

# Higher rank wins when one test case is written more than once in a report, for example
# pytest records a failing call and a failing teardown as two testcase elements.
STATUS_RANK: dict[Status, int] = {"passed": 0, "flaky": 1, "skipped": 2, "failed": 3, "errored": 4}


@dataclass(frozen=True)
class CaseResult:
    """The status of one test case and the report it came from."""

    status: Status
    report: Path


def _case_status(case: Element) -> Status:
    """Return the status of a JUnit testcase, where error outranks failure and skip.

    Gradle writes flakyFailure when a retried test fails and then passes. A required test
    has to pass outright, so that is reported as flaky rather than passed.
    """
    if case.find("error") is not None:
        return "errored"
    if case.find("failure") is not None:
        return "failed"
    if case.find("skipped") is not None:
        return "skipped"
    if case.find("flakyFailure") is not None or case.find("flakyError") is not None:
        return "flaky"
    return "passed"


def _python_id(case: Element, root: Path) -> str | None:
    """Rebuild the pytest node id of a testcase from its classname and name.

    pytest writes classname as the collecting module's path with dots for slashes, followed
    by any test classes. The longest dotted prefix that names a file under the project root
    is the module, and the rest are classes. The classname is used rather than the file
    attribute because an inherited test's file is the base class's module, not the module
    that collected it.
    """
    classname = case.get("classname")
    name = case.get("name")
    if not classname or not name:
        return None
    parts = classname.split(".")
    for split in range(len(parts), 0, -1):
        module = "/".join(parts[:split]) + ".py"
        if (root / module).is_file():
            return "::".join([module, *parts[split:], name])
    return None


def _java_id(case: Element, root: Path) -> str | None:
    """Build an id from a Gradle JUnit Platform testcase, dropping a trailing '()'.

    Only the empty parentheses of no-argument methods are dropped. Parameterized names keep
    the exact text Gradle writes.
    """
    classname = case.get("classname")
    name = case.get("name")
    if not classname or not name:
        return None
    return f"{classname}::{name.removesuffix('()')}"


def _ts_id(case: Element, root: Path) -> str | None:
    """Build an id from a Vitest JUnit testcase, whose classname is the test file path."""
    classname = case.get("classname")
    name = case.get("name")
    if not classname or not name:
        return None
    return f"{classname}::{name}"


ID_BUILDERS: dict[str, Callable[[Element, Path], str | None]] = {
    "py": _python_id,
    "java": _java_id,
    "ts": _ts_id,
}
LANGUAGES = tuple(ID_BUILDERS)


def _report_files(location: str) -> list[Path]:
    """Return the XML reports at a location, which may be a file, a directory or a glob."""
    files: list[Path] = []
    for match in sorted(glob.glob(location, recursive=True)):
        path = Path(match)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.xml") if p.is_file()))
        elif path.is_file():
            files.append(path)
    return files


def _read_report(report: Path, language: str, root: Path) -> tuple[list[tuple[str, Status]], str]:
    """Return the (id, status) pairs of one report and a problem message, empty when none."""
    try:
        tree_root = ElementTree.parse(report).getroot()
    except (OSError, ParseError, DefusedXmlException) as exc:
        return [], f"{report}: unreadable JUnit XML ({exc})"
    if tree_root is None:
        return [], f"{report}: unreadable JUnit XML (no root element)"

    build_id = ID_BUILDERS[language]
    cases: list[tuple[str, Status]] = []
    unmapped = 0
    for case in tree_root.iter("testcase"):
        case_id = build_id(case, root)
        if case_id is None:
            unmapped += 1
        else:
            cases.append((f"{language}::{case_id}", _case_status(case)))
    if unmapped:
        return cases, f"{report}: {unmapped} testcase(s) could not be mapped to a {language} id"
    return cases, ""


def _malformed(test_id: str) -> bool:
    """Return True when an id lacks a known language prefix or a name, or has stray spaces."""
    language, sep, rest = test_id.partition("::")
    return (
        language not in LANGUAGES
        or not sep
        or not rest
        or rest != rest.strip()
        or rest.endswith("::")
    )


def find_problems(
    required: list[str], reports: dict[str, list[str]], root: Path | None = None
) -> list[str]:
    """Return every reason the required ids do not all pass.

    Report-level problems come first: a language with no reports, a location with nothing
    in it, an unreadable report, testcases that cannot be mapped to ids. Then comes one
    message per required id that is malformed, missing, in more than one report, or did not
    pass. Only languages that appear in the required ids are read.
    """
    project_root = root or Path.cwd()
    problems: list[str] = []
    languages = sorted({t.partition("::")[0] for t in required if not _malformed(t)})

    results: dict[str, list[CaseResult]] = defaultdict(list)
    for language in languages:
        files_by_location = {loc: _report_files(loc) for loc in reports.get(language, [])}
        if not any(files_by_location.values()):
            problems.append(f"no JUnit reports found for language '{language}'")
            continue
        for location, files in files_by_location.items():
            if not files:
                problems.append(f"{location}: no JUnit reports at this location")
            for report in files:
                cases, issue = _read_report(report, language, project_root)
                if issue:
                    problems.append(issue)
                worst: dict[str, Status] = {}
                for case_id, status in cases:
                    if case_id not in worst or STATUS_RANK[status] > STATUS_RANK[worst[case_id]]:
                        worst[case_id] = status
                for case_id, status in worst.items():
                    results[case_id].append(CaseResult(status, report))

    for test_id in required:
        if _malformed(test_id):
            problems.append(
                f"{test_id!r}: id must be LANG::name with LANG one of {', '.join(LANGUAGES)}"
            )
            continue
        matches = results.get(test_id, [])
        if not matches:
            problems.append(f"{test_id}: not found in any report")
        elif len(matches) > 1:
            where = ", ".join(str(match.report) for match in matches)
            problems.append(f"{test_id}: found in {len(matches)} reports ({where})")
        elif matches[0].status != "passed":
            problems.append(f"{test_id}: {matches[0].status}")
    return problems


def load_required(path: Path) -> list[str]:
    """Read the required ids from a JSON object with a non-empty 'required_tests' list."""
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: cannot read required tests ({exc})") from exc
    if not isinstance(data, dict) or "required_tests" not in data:
        raise SystemExit(f"{path}: expected a JSON object with a 'required_tests' list")
    required = data["required_tests"]
    if not isinstance(required, list) or not all(isinstance(t, str) for t in required):
        raise SystemExit(f"{path}: 'required_tests' must be a list of strings")
    if not required:
        raise SystemExit(f"{path}: 'required_tests' is empty, so there is nothing to verify")
    return required


def _parse_report_args(values: list[str]) -> dict[str, list[str]]:
    """Turn LANG=LOCATION arguments into a mapping of language to report locations."""
    reports: dict[str, list[str]] = defaultdict(list)
    for value in values:
        language, sep, location = value.partition("=")
        if not sep or language not in LANGUAGES or not location:
            raise SystemExit(
                f"--report expects LANG=LOCATION with LANG one of {', '.join(LANGUAGES)}, "
                f"got '{value}'"
            )
        reports[language].append(location)
    return reports


def main(argv: list[str] | None = None) -> int:
    """Verify the required ids listed in a JSON file against the given reports."""
    parser = argparse.ArgumentParser(description="Check required test ids against JUnit XML.")
    parser.add_argument(
        "--required",
        type=Path,
        required=True,
        help="JSON file whose 'required_tests' key lists the required ids",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="LANG=LOCATION",
        help="JUnit XML file, directory or glob for a language, repeatable",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="project root that pytest module paths are relative to",
    )
    args = parser.parse_args(argv)

    try:
        required = load_required(args.required)
        reports = _parse_report_args(args.report)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    problems = find_problems(required, reports, args.root)
    if problems:
        print(f"{len(problems)} problem(s) with required tests:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"All {len(required)} required tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
