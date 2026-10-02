#!/bin/bash
# Filename matches class check - ensures filenames match their primary class

echo "→ Running filename matches class check..."
.venv/bin/python scripts/check_filename_matches_class.py src/ examples/ tests/ scripts/ 2>&1
exit $?
