#!/bin/bash
# Ruff format check

echo "→ Running ruff format..."
.venv/bin/ruff format src/ examples/ tests/ scripts/ 2>&1
exit $?
