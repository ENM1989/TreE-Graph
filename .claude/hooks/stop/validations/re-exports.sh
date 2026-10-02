#!/bin/bash
# Re-export check - ensures no re-exports

echo "→ Running re-export check..."
.venv/bin/python scripts/check_no_reexports.py src/ examples/ tests/ scripts/ 2>&1
exit $?
