#!/bin/bash
# Ruff check - linting

echo "→ Running ruff check..."
.venv/bin/ruff check src/ examples/ tests/ scripts/ 2>&1
exit $?
