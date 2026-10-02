#!/bin/bash
# Pytest with coverage (excluding slow tests)

echo "→ Running pytest with coverage (excluding slow tests)..."
.venv/bin/pytest tests/ -m "not slow" --tb=short --cov=src --cov-report=term-missing --cov-fail-under=80 2>&1
exit $?
