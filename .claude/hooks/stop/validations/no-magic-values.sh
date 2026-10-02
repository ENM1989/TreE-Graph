#!/bin/bash
# No magic values check - ensures all numbers and strings are defined as constants

echo "→ Running no magic values check..."
.venv/bin/python scripts/check_no_magic_values.py src/ examples/ tests/ scripts/ 2>&1
exit $?