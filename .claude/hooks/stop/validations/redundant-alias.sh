#!/bin/bash
# Redundant import alias check - ensures no redundant aliases in imports

echo "→ Running redundant import alias check..."
.venv/bin/python scripts/check_redundant_alias.py src/ examples/ tests/ scripts/ 2>&1
exit $?
