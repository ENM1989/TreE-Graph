#!/bin/bash
# No union list return check - ensures functions don't return list with union types

echo "→ Running no union list return check..."
.venv/bin/python scripts/check_no_union_list_return.py src/ examples/ tests/ scripts/ 2>&1
exit $?