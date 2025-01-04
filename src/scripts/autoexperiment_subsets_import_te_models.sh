#!/bin/bash

# Default values
te_model_path_list_file=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --timestamp) timestamp_flag=true ;;                     # Enable timestamp
        --te-model-path-list-file) te_model_path_list_file="$2"; shift ;;  # Handle space-separated value
        --te-model-path-list-file=*) te_model_path_list_file="${1#*=}" ;;  # Handle "="-separated value
        *) echo "Unknown option: $1" && exit 1 ;;               # Handle invalid options
    esac
    shift
done

# Ensure --te-model-path-list-file is provided
if [[ -z "$te_model_path_list_file" ]]; then
    echo "Error: --te-model-path-list-file is required."
    exit 1
fi

# Check if the --timestamp flag is passed
if [[ "$timestamp_flag" == true ]]; then
    # Generate a timestamp
    timestamp=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y_%m_%d_at_%H_%M_%S'))")
    
    # Run with timestamp
    python3 src/scripts/autoexperiment_subsets_import_te_models.py \
        --timestamp=$timestamp \
        --called-from-bash-script=True \
        --te-model-path-list-file=$te_model_path_list_file | tee \
        ./run_terminal_logs/autoexperiment_subsets_import_te_models/autoexperiment_subsets_import_te_models_output_log_$timestamp.log
else
    # Run without timestamp, explicitly passing "None"
    python3 src/scripts/autoexperiment_subsets_import_te_models.py \
        --timestamp=None \
        --called-from-bash-script=True \
        --te-model-path-list-file=$te_model_path_list_file
fi
