#!/bin/bash
# No constants file check - ensures constants are organized as packages, not single files

echo "→ Running no constants file check..."
.venv/bin/python scripts/check_no_constants_file.py src/ examples/ tests/ scripts/ 2>&1
exit $?