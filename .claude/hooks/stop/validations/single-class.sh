#!/bin/bash
# Single-class check - ensures each file contains only one class

echo "→ Running single-class check..."
.venv/bin/python scripts/check_single_class.py src/ examples/ scripts/ 2>&1
exit $?
