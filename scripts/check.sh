#!/usr/bin/env bash
# Lint, format, type and test gate, run identically locally and in CI.
# Every step runs even after a failure so one pass shows all problems, and the script exits
# non-zero if any step failed. The Java and TypeScript steps run only when eqd-pricing/ and
# desk-ui/ exist. Old JUnit reports are deleted first, so every report left afterwards comes
# from this run: build/test-results/python.xml, eqd-pricing/*/build/test-results/ and
# desk-ui/test-results/.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

if ! rm -rf build/test-results eqd-pricing/*/build/test-results desk-ui/test-results ||
    ! mkdir -p build/test-results; then
    echo "Cannot clear old JUnit reports"
    exit 1
fi

failed=()

run_step() {
    local label="$1"
    shift
    echo "Running $label"
    if ! "$@"; then
        failed+=("$label")
    fi
}

# Records a step that could not run because its project is incomplete.
missing_step() {
    echo "Cannot run $1: $2"
    failed+=("$1")
}

run_step "ruff check" uv run ruff check .
run_step "ruff format" uv run ruff format --check .
run_step "mypy" uv run mypy
run_step "pytest" uv run pytest --junitxml=build/test-results/python.xml

if [ -d eqd-pricing ]; then
    if [ -x eqd-pricing/gradlew ]; then
        run_step "gradle check" eqd-pricing/gradlew -p eqd-pricing check
    else
        missing_step "gradle check" "eqd-pricing/gradlew is missing or not executable"
    fi
fi

if [ -d desk-ui ]; then
    if [ -f desk-ui/package.json ]; then
        run_step "desk-ui lint" npm --prefix desk-ui run lint
        run_step "desk-ui typecheck" npm --prefix desk-ui run typecheck
        run_step "desk-ui test" npm --prefix desk-ui test
    else
        missing_step "desk-ui checks" "desk-ui/package.json is missing"
    fi
fi

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed steps: ${failed[*]}"
    exit 1
fi
echo "All checks passed."
