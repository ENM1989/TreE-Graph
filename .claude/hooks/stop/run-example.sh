#!/bin/bash
# Session-stop hook - runs the examples script when the Claude session ends

echo "Running examples/basic_simplification.py..."
.venv/bin/python examples/basic_simplification.py

if [ $? -ne 0 ]; then
    echo "WARNING: examples/basic_simplification.py failed"
fi
