#!/bin/bash
# No bare imports check - ensures no bare imports in src/ and tests/

echo "→ Running no bare imports check..."
.venv/bin/python scripts/check_no_bare_imports.py src/ examples/ tests/ 2>&1
exit $?
