#!/bin/bash
# Test class naming check - ensures test classes are named after the class they test

echo "→ Running test class naming check..."
.venv/bin/python scripts/check_test_class_naming.py tests/ 2>&1
exit $?
