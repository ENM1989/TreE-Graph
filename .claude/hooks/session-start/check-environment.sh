#!/bin/bash
# Session-start hook - check Python virtual environment availability

# Check if virtual environment exists and can be activated
if [ ! -d ".venv" ]; then
    echo "WARNING: Python virtual environment (.venv) not found!"
    echo ""
    echo "To create and activate the virtual environment, run:"
    echo "  python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
    echo ""
    echo "Many commands will fail without the virtual environment."
    echo ""
elif [ ! -f ".venv/bin/activate" ]; then
    echo "WARNING: Python virtual environment exists but activation script is missing!"
    echo "The .venv directory may be corrupted or incomplete."
    echo ""
    echo "To recreate the virtual environment, run:"
    echo "  rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
    echo ""
else
    # Try to actually activate the venv
    source .venv/bin/activate 2>/dev/null
    ACTIVATION_EXIT=$?

    if [ $ACTIVATION_EXIT -ne 0 ]; then
        echo "WARNING: Failed to activate Python virtual environment!"
        echo ""
        echo "The activation script returned an error. Try recreating the venv:"
        echo "  rm -rf .venv && python -m venv .venv && source .venv/bin/activate && pip install -e '.[dev]'"
        echo ""
    fi
fi
