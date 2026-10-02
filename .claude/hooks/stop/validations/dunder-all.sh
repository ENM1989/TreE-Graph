#!/bin/bash
# __all__ check - ensures no __all__ definitions

echo "→ Running __all__ check..."
.venv/bin/python scripts/check_no_dunder_all.py src/ examples/ tests/ scripts/ 2>&1
exit $?
