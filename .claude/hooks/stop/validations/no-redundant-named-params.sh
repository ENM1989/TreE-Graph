#!/bin/bash
# No redundant named parameters check - ensures functions don't use param=param style

echo "→ Running no redundant named parameters check..."
.venv/bin/python scripts/check_no_redundant_named_params.py src/ examples/ tests/ scripts/ 2>&1
exit $?