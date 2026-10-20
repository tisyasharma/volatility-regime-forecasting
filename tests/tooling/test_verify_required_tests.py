"""Tests for the required-test verifier."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.verify_required_tests import LANGUAGES, find_problems, main

REPO_ROOT = Path(__file__).resolve().parents[2]

JAVA_REPORT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.pricing.BlackScholesTest" tests="6" skipped="1" failures="1">
  <testcase name="putCallParity()" classname="com.example.pricing.BlackScholesTest"/>
  <testcase name="vegaIsPositive" classname="com.example.pricing.BlackScholesTest"/>
  <testcase name="deltaBounds()" classname="com.example.pricing.BlackScholesTest">
    <skipped/>
  </testcase>
  <testcase name="gammaSymmetry()" classname="com.example.pricing.BlackScholesTest">
    <failure message="expected 0.5">AssertionError</failure>
  </testcase>
  <testcase name="[1] 0.5" classname="com.example.pricing.BlackScholesTest$Strikes"/>
  <testcase name="thetaSign()" classname="com.example.pricing.BlackScholesTest">
    <flakyFailure message="first run">AssertionError</flakyFailure>
  </testcase>
</testsuite>
"""

TS_REPORT = """<?xml version="1.0" encoding="UTF-8" ?>
<testsuites name="vitest tests" tests="2" failures="0" errors="1">
  <testsuite name="src/blotter.test.ts" tests="2">
    <testcase classname="src/blotter.test.ts" name="blotter &gt; sorts by vrp"/>
    <testcase classname="src/blotter.test.ts" name="blotter &gt; renders greeks">
      <error message="boom">TypeError</error>
    </testcase>
  </testsuite>
</testsuites>
"""

BASE_TESTS = '''
class SharedChecks:
    def test_shared(self):
        """Inherited by test classes in other modules."""
        assert True
'''

SAMPLE_TESTS = '''
import pytest

from tests.base_checks import SharedChecks


def test_passes():
    """Always passes."""
    assert True


def test_fails():
    """Always fails."""
    assert False


@pytest.mark.skip(reason="skipped on purpose")
def test_skipped():
    """Always skipped."""


@pytest.mark.xfail(reason="expected to fail")
def test_xfail():
    """Fails as expected."""
    assert False


@pytest.mark.xfail(reason="expected to fail", strict=False)
def test_xpass():
    """Passes although marked xfail."""
    assert True


@pytest.fixture
def broken_teardown():
    """Raises during teardown."""
    yield
    raise RuntimeError("teardown failed")


def test_passes_then_teardown_errors(broken_teardown):
    """The call passes but teardown errors."""
    assert True


def test_fails_then_teardown_errors(broken_teardown):
    """The call fails and teardown errors as well."""
    assert False


@pytest.mark.parametrize("value", [1, 2])
def test_param(value):
    """Passes for each parameter."""
    assert value > 0


@pytest.mark.parametrize("value", ["a::b", "x.y", "p[q]"])
def test_awkward_param(value):
    """Parameters containing separators still map to their ids."""
    assert value


class TestGroup:
    def test_inside_class(self):
        """Passes inside a test class."""
        assert True

    class TestNested:
        def test_inside_nested(self):
            """Passes inside a nested class."""
            assert True


class TestInheritsShared(SharedChecks):
    """Collects test_shared from another module."""
'''


@pytest.fixture(scope="module")
def python_project(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Run real pytest on a small suite and return the project root and its JUnit report."""
    root = tmp_path_factory.mktemp("pyproject")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "base_checks.py").write_text(textwrap.dedent(BASE_TESTS))
    (root / "tests" / "test_sample.py").write_text(textwrap.dedent(SAMPLE_TESTS))
    report = root / "python.xml"
    # The sample suite fails on purpose, so the outcome is checked through the summary line.
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={report}", "tests"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    summary = run.stdout.strip().splitlines()[-1]
    for count in ("2 failed", "10 passed", "1 skipped", "1 xfailed", "1 xpassed", "2 error"):
        assert count in summary, summary
    return root, report


def _py(test_name: str) -> str:
    """Return the required id of a test in the sample module."""
    return f"py::tests/test_sample.py::{test_name}"


def _py_problems(project: tuple[Path, Path], required: list[str]) -> list[str]:
    """Run find_problems against the sample pytest report."""
    root, report = project
    return find_problems(required, {"py": [str(report)]}, root)


@pytest.fixture
def java_project(tmp_path: Path) -> Path:
    """Write a Gradle-style report under a module build directory and return the project root."""
    directory = tmp_path / "eqd-pricing" / "pricing-core" / "build" / "test-results" / "test"
    directory.mkdir(parents=True)
    (directory / "TEST-com.example.pricing.BlackScholesTest.xml").write_text(JAVA_REPORT)
    return tmp_path / "eqd-pricing"


@pytest.fixture
def ts_report(tmp_path: Path) -> Path:
    """Write a Vitest-style JUnit report and return its path."""
    report = tmp_path / "vitest.xml"
    report.write_text(TS_REPORT)
    return report


def test_passing_python_ids_have_no_problems(python_project: tuple[Path, Path]) -> None:
    """Plain, parametrized, class, nested-class and inherited pytest ids all map when passing."""
    required = [
        _py("test_passes"),
        _py("test_param[1]"),
        _py("test_param[2]"),
        _py("test_awkward_param[a::b]"),
        _py("test_awkward_param[x.y]"),
        _py("test_awkward_param[p[q]]"),
        _py("TestGroup::test_inside_class"),
        _py("TestGroup::TestNested::test_inside_nested"),
        _py("TestInheritsShared::test_shared"),
    ]
    assert _py_problems(python_project, required) == []


def test_python_non_passing_statuses_are_problems(python_project: tuple[Path, Path]) -> None:
    """Failed, skipped and xfailed tests are problems, and a non-strict xpass counts as passed."""
    required = [_py("test_fails"), _py("test_skipped"), _py("test_xfail"), _py("test_xpass")]
    assert _py_problems(python_project, required) == [
        f"{_py('test_fails')}: failed",
        f"{_py('test_skipped')}: skipped",
        f"{_py('test_xfail')}: skipped",
    ]


def test_teardown_error_takes_the_worst_status(python_project: tuple[Path, Path]) -> None:
    """pytest writes a teardown error as a second testcase, which merges into one result."""
    required = [_py("test_passes_then_teardown_errors"), _py("test_fails_then_teardown_errors")]
    assert _py_problems(python_project, required) == [
        f"{_py('test_passes_then_teardown_errors')}: errored",
        f"{_py('test_fails_then_teardown_errors')}: errored",
    ]


def test_missing_id_is_a_problem(python_project: tuple[Path, Path]) -> None:
    """A required id that never ran is reported as not found."""
    assert _py_problems(python_project, [_py("test_absent")]) == [
        f"{_py('test_absent')}: not found in any report"
    ]


def test_python_case_outside_the_root_cannot_be_mapped(
    python_project: tuple[Path, Path], tmp_path: Path
) -> None:
    """Checking a report against the wrong project root names the report instead of guessing."""
    _, report = python_project
    problems = find_problems([_py("test_passes")], {"py": [str(report)]}, tmp_path)
    assert problems[0].startswith(f"{report}: ")
    assert problems[0].endswith("testcase(s) could not be mapped to a py id")
    assert problems[1:] == [f"{_py('test_passes')}: not found in any report"]


def test_java_ids_drop_only_empty_parentheses(java_project: Path) -> None:
    """Gradle's '()' on no-argument methods is dropped, other names and nested classes are kept."""
    required = [
        "java::com.example.pricing.BlackScholesTest::putCallParity",
        "java::com.example.pricing.BlackScholesTest::vegaIsPositive",
        "java::com.example.pricing.BlackScholesTest$Strikes::[1] 0.5",
    ]
    assert find_problems(required, {"java": [str(java_project)]}) == []
    with_parentheses = "java::com.example.pricing.BlackScholesTest::putCallParity()"
    assert find_problems([with_parentheses], {"java": [str(java_project)]}) == [
        f"{with_parentheses}: not found in any report"
    ]


def test_java_skipped_failed_and_flaky_are_problems(java_project: Path) -> None:
    """Skipped, failed and flaky Java cases are reported from a nested report directory."""
    required = [
        "java::com.example.pricing.BlackScholesTest::deltaBounds",
        "java::com.example.pricing.BlackScholesTest::gammaSymmetry",
        "java::com.example.pricing.BlackScholesTest::thetaSign",
    ]
    assert find_problems(required, {"java": [str(java_project)]}) == [
        "java::com.example.pricing.BlackScholesTest::deltaBounds: skipped",
        "java::com.example.pricing.BlackScholesTest::gammaSymmetry: failed",
        "java::com.example.pricing.BlackScholesTest::thetaSign: flaky",
    ]


def test_glob_location_reads_only_test_results(java_project: Path) -> None:
    """A glob over build/test-results ignores XML elsewhere, such as test resources."""
    resources = java_project / "pricing-core" / "src" / "test" / "resources"
    resources.mkdir(parents=True)
    (resources / "fixture.xml").write_text(
        '<testsuite><testcase classname="com.example.Fake" name="planted()"/></testsuite>'
    )
    planted = "java::com.example.Fake::planted"
    location = str(java_project / "*" / "build" / "test-results")
    assert find_problems([planted], {"java": [location]}) == [f"{planted}: not found in any report"]
    assert find_problems([planted], {"java": [str(java_project)]}) == []


def test_ts_passed_and_errored(ts_report: Path) -> None:
    """Vitest ids use the file and the full suite path, and an errored case is a problem."""
    required = [
        "ts::src/blotter.test.ts::blotter > sorts by vrp",
        "ts::src/blotter.test.ts::blotter > renders greeks",
    ]
    assert find_problems(required, {"ts": [str(ts_report)]}) == [
        "ts::src/blotter.test.ts::blotter > renders greeks: errored"
    ]


def test_id_in_two_reports_is_a_problem(java_project: Path, tmp_path: Path) -> None:
    """The same id in two reports is ambiguous, so it fails even when both copies passed."""
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "TEST-copy.xml").write_text(JAVA_REPORT)
    test_id = "java::com.example.pricing.BlackScholesTest::putCallParity"
    problems = find_problems([test_id], {"java": [str(java_project), str(copy)]})
    assert len(problems) == 1
    assert problems[0].startswith(f"{test_id}: found in 2 reports (")
    assert "TEST-copy.xml" in problems[0]
    assert "TEST-com.example.pricing.BlackScholesTest.xml" in problems[0]


def test_unreadable_reports_are_named_problems(tmp_path: Path, ts_report: Path) -> None:
    """Empty, truncated, entity-bearing and unreadable reports are named, and others still count."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "empty.xml").write_text("")
    (reports / "truncated.xml").write_text("<testsuite><testcase name='x'")
    (reports / "entities.xml").write_text(
        '<?xml version="1.0"?><!DOCTYPE t [<!ENTITY e "boom">]><testsuite>&e;</testsuite>'
    )
    (reports / "unreadable.xml").write_text("<testsuite/>")
    (reports / "unreadable.xml").chmod(0)
    (reports / "folder.xml").mkdir()
    passing = "ts::src/blotter.test.ts::blotter > sorts by vrp"
    try:
        problems = find_problems([passing], {"ts": [str(reports), str(ts_report)]})
    finally:
        (reports / "unreadable.xml").chmod(0o644)
    assert len(problems) == 4
    for name in ("empty.xml", "entities.xml", "truncated.xml", "unreadable.xml"):
        assert any(p.startswith(f"{reports / name}: unreadable JUnit XML") for p in problems)


def test_language_without_reports_is_a_problem(tmp_path: Path) -> None:
    """Required ids for a language with no reports fail instead of being ignored."""
    empty = tmp_path / "empty"
    empty.mkdir()
    test_id = "ts::src/blotter.test.ts::blotter > sorts by vrp"
    expected = [
        "no JUnit reports found for language 'ts'",
        f"{test_id}: not found in any report",
    ]
    assert find_problems([test_id], {"ts": [str(tmp_path / "absent")]}) == expected
    assert find_problems([test_id], {"ts": [str(empty)]}) == expected
    assert find_problems([test_id], {}) == expected


def test_empty_location_beside_a_real_one_is_a_problem(ts_report: Path, tmp_path: Path) -> None:
    """Every listed location must hold reports, even when another one for the language does."""
    absent = tmp_path / "absent"
    problems = find_problems(
        ["ts::src/blotter.test.ts::blotter > sorts by vrp"],
        {"ts": [str(ts_report), str(absent)]},
    )
    assert problems == [f"{absent}: no JUnit reports at this location"]


def test_unused_languages_are_not_read(python_project: tuple[Path, Path], tmp_path: Path) -> None:
    """A broken report for a language no required id uses does not affect the result."""
    broken = tmp_path / "broken.xml"
    broken.write_text("<testsuite")
    root, report = python_project
    reports = {"py": [str(report)], "java": [str(broken)]}
    assert find_problems([_py("test_passes")], reports, root) == []


@pytest.mark.parametrize(
    "bad_id",
    [
        "tests/test_sample.py::test_passes",
        "rs::src/lib.rs::works",
        "py::",
        "py:: tests/test_sample.py::test_passes",
        "py::tests/test_sample.py::test_passes ",
        "java::com.example.Foo::",
    ],
)
def test_malformed_ids_are_problems(python_project: tuple[Path, Path], bad_id: str) -> None:
    """Ids without a known language or a name, or with stray spaces, are rejected."""
    assert _py_problems(python_project, [bad_id]) == [
        f"{bad_id!r}: id must be LANG::name with LANG one of py, java, ts"
    ]


def _write_required(path: Path, payload: object) -> Path:
    """Write a required-tests JSON file and return its path."""
    path.write_text(json.dumps(payload))
    return path


def test_main_exit_codes(python_project: tuple[Path, Path], tmp_path: Path) -> None:
    """main returns 0 when every required id passed and 1 when any did not."""
    root, report = python_project
    args = ["--report", f"py={report}", "--root", str(root)]
    passing = _write_required(tmp_path / "pass.json", {"required_tests": [_py("test_passes")]})
    failing = _write_required(tmp_path / "fail.json", {"required_tests": [_py("test_fails")]})
    assert main(["--required", str(passing), *args]) == 0
    assert main(["--required", str(failing), *args]) == 1


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "cannot read required tests"),
        ("{not json", "cannot read required tests"),
        (json.dumps({"required_test": ["py::a"]}), "expected a JSON object"),
        (json.dumps(["py::a"]), "expected a JSON object"),
        (json.dumps({"required_tests": None}), "must be a list of strings"),
        (json.dumps({"required_tests": "py::a"}), "must be a list of strings"),
        (json.dumps({"required_tests": [1]}), "must be a list of strings"),
        (json.dumps({"required_tests": []}), "is empty"),
        (b"\xff\xfe not utf-8", "cannot read required tests"),
    ],
)
def test_unusable_required_file_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], contents: str | bytes | None, message: str
) -> None:
    """A missing, misspelled, mistyped, undecodable or empty required list is an error."""
    path = tmp_path / "required.json"
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    elif contents is not None:
        path.write_text(contents)
    assert main(["--required", str(path)]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("bad_report", ["build/python.xml", "rs=build/x.xml", "py="])
def test_bad_report_argument_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad_report: str
) -> None:
    """A --report value that is not LANG=LOCATION with a known language is an error."""
    path = _write_required(tmp_path / "required.json", {"required_tests": ["py::tests/x.py::t"]})
    assert main(["--required", str(path), "--report", bad_report]) == 2
    assert "LANG=LOCATION" in capsys.readouterr().err


def test_script_runs_as_a_command(python_project: tuple[Path, Path], tmp_path: Path) -> None:
    """The verifier runs as a standalone script, the way the gate calls it."""
    root, report = python_project
    path = _write_required(tmp_path / "required.json", {"required_tests": [_py("test_passes")]})
    run = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_required_tests.py"),
            "--required",
            str(path),
            "--report",
            f"py={report}",
            "--root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "All 1 required tests passed."


STATUS_ELEMENTS = {
    "passed": "",
    "skipped": "<skipped/>",
    "failed": "<failure/>",
    "errored": "<error/>",
    "flaky": "<flakyFailure/>",
}
CASE_NAMES = st.from_regex(
    r"case[a-zA-Z0-9]{0,8}( > [a-z]{1,5})?(\[[a-z.:]{1,6}\])?", fullmatch=True
)


@given(
    language=st.sampled_from(["java", "ts"]),
    cases=st.dictionaries(
        keys=CASE_NAMES, values=st.sampled_from(sorted(STATUS_ELEMENTS)), max_size=12
    ),
    absent=st.lists(st.from_regex(r"absent[0-9]{1,3}", fullmatch=True), max_size=3, unique=True),
)
def test_problems_are_exactly_the_cases_that_did_not_pass(
    tmp_path_factory: pytest.TempPathFactory,
    language: str,
    cases: dict[str, str],
    absent: list[str],
) -> None:
    """For any mix of statuses and names, the problems are the non-passing and missing ids."""
    classname = "pkg.SomeTest" if language == "java" else "src/some.test.ts"
    rows = "".join(
        f'<testcase name="{name}" classname="{classname}">{STATUS_ELEMENTS[status]}</testcase>'
        for name, status in cases.items()
    )
    report = tmp_path_factory.mktemp("report") / "report.xml"
    report.write_text(f"<testsuites><testsuite>{rows}</testsuite></testsuites>")
    required = [f"{language}::{classname}::{name}" for name in [*cases, *absent]]

    expected = [
        f"{language}::{classname}::{name}: {status}"
        for name, status in cases.items()
        if status != "passed"
    ] + [f"{language}::{classname}::{name}: not found in any report" for name in absent]
    assert find_problems(required, {language: [str(report)]}) == expected


def test_languages_match_the_documented_prefixes() -> None:
    """The prefixes the module docstring documents are exactly the languages with id builders."""
    assert LANGUAGES == ("py", "java", "ts")
