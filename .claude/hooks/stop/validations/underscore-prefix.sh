#!/bin/bash
# Underscore prefix check - ensures private members use underscore prefix

echo "→ Running underscore prefix check..."
.venv/bin/python scripts/check_underscore_prefix.py src/ examples/ tests/ scripts/ 2>&1
exit $?
