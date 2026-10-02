#!/bin/bash
# Constants in classes check - ensures constants are defined inside classes

echo "→ Running constants in classes check..."
.venv/bin/python scripts/check_constants_in_classes.py src/ examples/ tests/ scripts/ 2>&1
exit $?