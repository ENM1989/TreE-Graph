#!/bin/bash
# Post-task validation hook - runs static analysis and tests after coding tasks

# Read input to check if this is a recursive call
INPUT=$(cat)
if echo "$INPUT" | grep -q '"hook_active":true'; then
    exit 0
fi

# Run all static analysis tools and tests
echo "Running post-task validation..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATIONS_DIR="$SCRIPT_DIR/validations"

# Track failures for summary
declare -a FAILED_VALIDATIONS=()

# Run each validation script and track exit codes
run_validation() {
    local name="$1"
    local script="$VALIDATIONS_DIR/$name"

    if [ -x "$script" ]; then
        "$script"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            FAILED_VALIDATIONS+=("$name (exit code: $exit_code)")
        fi
        return $exit_code
    else
        echo "ERROR: Validation script not found or not executable: $script"
        FAILED_VALIDATIONS+=("$name (script not found)")
        return 1
    fi
}

# Custom validations (Python scripts)
run_validation "single-class.sh"
run_validation "standalone-functions.sh"
run_validation "underscore-prefix.sh"
run_validation "dunder-all.sh"
run_validation "redundant-alias.sh"
run_validation "re-exports.sh"
run_validation "filename-matches-class.sh"
run_validation "no-bare-imports.sh"
run_validation "imports-separate-lines.sh"
run_validation "no-static-methods.sh"
run_validation "no-union-list-return.sh"
run_validation "no-heterogeneous-tuple-return.sh"
run_validation "no-redundant-named-params.sh"
run_validation "no-single-char-vars.sh"
run_validation "test-class-naming.sh"
run_validation "constants-in-classes.sh"
run_validation "no-constants-file.sh"

# Linting and formatting
run_validation "ruff-format.sh"
run_validation "ruff-check.sh"

# Type checking
run_validation "mypy.sh"
run_validation "pyright.sh"

# Tests
run_validation "pytest.sh"

# Examples
run_validation "run-example.sh"

# Check results
if [ ${#FAILED_VALIDATIONS[@]} -ne 0 ]; then
    echo ""
    echo "VALIDATION FAILED:"
    for validation in "${FAILED_VALIDATIONS[@]}"; do
        echo "  - $validation"
    done
    echo ""
    echo "Please fix the issues above before completing the task."
    exit 1
fi

echo "All validations passed!"
exit 0
