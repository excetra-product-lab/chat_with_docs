#!/bin/bash

# RAG Evaluation Entry Point
# This script provides easy access to the evaluation system from the project root

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pass all arguments to the actual evaluation script
exec "$SCRIPT_DIR/scripts/run_evaluation.sh" "$@"
