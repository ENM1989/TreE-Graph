#!/bin/bash
# MyPy type checking

echo "→ Running mypy..."
.venv/bin/mypy src/ examples/ tests/ scripts/ 2>&1
exit $?
