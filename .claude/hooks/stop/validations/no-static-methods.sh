#!/bin/bash
# No static methods check - ensures no @staticmethod decorators are used

echo "→ Running no static methods check..."
.venv/bin/python scripts/check_no_static_methods.py src/ examples/ tests/ scripts/ 2>&1
exit $?
