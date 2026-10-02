#!/bin/bash
# Check for single character variable names

echo "→ Running no single character variable names check..."
.venv/bin/python scripts/check_no_single_char_vars.py src/ examples/ tests/ scripts/ 2>&1
exit $?