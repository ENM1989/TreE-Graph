#!/bin/bash
# Standalone functions check - ensures src/ has no standalone functions

echo "→ Running standalone functions check..."
.venv/bin/python scripts/check_no_standalone_functions.py src/ 2>&1
exit $?
