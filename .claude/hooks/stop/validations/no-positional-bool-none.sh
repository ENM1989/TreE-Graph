#!/bin/bash
# No positional bool/None check - ensures None, True, False use named parameters

echo "→ Running no positional bool/None check..."
.venv/bin/python scripts/check_no_positional_bool_none.py src/ examples/ tests/ scripts/ 2>&1
exit $?