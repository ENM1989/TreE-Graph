#!/bin/bash
# Imports separate lines check - ensures imports are on separate lines

echo "→ Running imports separate lines check..."
.venv/bin/python scripts/check_imports_separate_lines.py src/ examples/ tests/ scripts/ 2>&1
exit $?
