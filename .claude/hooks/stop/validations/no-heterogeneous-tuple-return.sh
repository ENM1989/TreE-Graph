#!/bin/bash
# No heterogeneous tuple return check - ensures functions don't return tuples with multiple types

echo "→ Running no heterogeneous tuple return check..."
.venv/bin/python scripts/check_no_heterogeneous_tuple_return.py src/ examples/ tests/ scripts/ 2>&1
exit $?