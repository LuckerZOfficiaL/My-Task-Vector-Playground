#!/bin/bash

# Function to get the current timestamp in the desired format
get_timestamp() {
  date +"%Y_%m_%d_at_%H_%M_%S"
}

# Parse arguments
TIMESTAMP=""
PYTHON_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--timestamp" ]]; then
    TIMESTAMP=$(get_timestamp)
  else
    PYTHON_ARGS+=("$arg")
  fi
done

# Add --called-from-bash argument
PYTHON_ARGS+=("--called-from-bash")

# Add timestamp to Python arguments if provided
if [[ -n "$TIMESTAMP" ]]; then
  PYTHON_ARGS+=("--timestamp" "$TIMESTAMP")
fi

# Build the command
CMD="python3 src/scripts/autoexperiment.py ${PYTHON_ARGS[@]}"

# Execute the command
if [[ -n "$TIMESTAMP" ]]; then
  # If --timestamp is provided, log output to a file
  LOG_FILE="logs/${TIMESTAMP}.log"
  mkdir -p logs
  $CMD 2>&1 | tee "$LOG_FILE"
else
  # Otherwise, just run the command
  $CMD
fi
