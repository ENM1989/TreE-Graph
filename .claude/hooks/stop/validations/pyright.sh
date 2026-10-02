#!/bin/bash
# Pyright type checking

echo "→ Running pyright..."
.venv/bin/pyright src/ examples/ tests/ scripts/ 2>&1
exit $?
