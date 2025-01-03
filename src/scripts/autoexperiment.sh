#!/bin/bash

# Check if the "--timestamp" flag is passed
if [[ "$1" == "--timestamp" ]]; then
    # Generate a timestamp
    timestamp=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y_%m_%d_at_%H_%M_%S'))")
    
    # Run with timestamp
    python3 src/scripts/autoexperiment.py --timestamp=$timestamp --called-from-bash-script=True | tee ./run_terminal_logs/autoexperiment_output_log_$timestamp.log
else
    # Run without timestamp, explicitly passing "None"
    python3 src/scripts/autoexperiment.py --timestamp=None --called-from-bash-script=True
fi
