#!/bin/bash

# Script to run the FlagFrenzy environment with the correct Python library path

# Activate the virtual environment
source .venv/bin/activate

# Set the library path for the SimulationInterface.cpython-39-darwin.so
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/python@3.9/3.9.22/Frameworks/Python.framework/Versions/3.9/lib

# Run the specified script or command
if [ $# -eq 0 ]; then
  # If no arguments provided, run the test environment
  python test_env.py
else
  # Otherwise run the specified command
  python "$@"
fi