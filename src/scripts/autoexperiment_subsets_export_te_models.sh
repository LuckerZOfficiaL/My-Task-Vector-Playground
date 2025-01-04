#!/bin/bash

# Default values
subset_len=""
num_subsets=""
subsets_idx_start=""
subsets_idx_end=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --timestamp) timestamp_flag=true ;;                     # Enable timestamp
        --subset-len) subset_len="$2"; shift ;;                 # Handle space-separated value
        --subset-len=*) subset_len="${1#*=}" ;;                 # Handle "="-separated value
        --num-subsets) num_subsets="$2"; shift ;;               # Handle space-separated value
        --num-subsets=*) num_subsets="${1#*=}" ;;               # Handle "="-separated value
        --subsets-idx-start) subsets_idx_start="$2"; shift ;;   # Handle space-separated value
        --subsets-idx-start=*) subsets_idx_start="${1#*=}" ;;   # Handle "="-separated value
        --subsets-idx-end) subsets_idx_end="$2"; shift ;;       # Handle space-separated value
        --subsets-idx-end=*) subsets_idx_end="${1#*=}" ;;       # Handle "="-separated value
        *) echo "Unknown option: $1" && exit 1 ;;               # Handle invalid options
    esac
    shift
done

# Ensure required arguments are provided
if [[ -z "$subset_len" ]]; then
    echo "Error: --subset-len is required."
    exit 1
fi

if [[ -z "$num_subsets" ]]; then
    echo "Error: --num-subsets is required."
    exit 1
fi

if [[ -z "$subsets_idx_start" ]]; then
    echo "Error: --subsets-idx-start is required."
    exit 1
fi

if [[ -z "$subsets_idx_end" ]]; then
    echo "Error: --subsets-idx-end is required."
    exit 1
fi

# Check if the --timestamp flag is passed
if [[ "$timestamp_flag" == true ]]; then
    # Generate a timestamp
    timestamp=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y_%m_%d_at_%H_%M_%S'))")
    
    # Run with timestamp
    python3 src/scripts/autoexperiment_subsets_export_te_models.py \
        --timestamp=$timestamp \
        --called-from-bash-script=True \
        --subset-len=$subset_len \
        --num-subsets=$num_subsets \
        --subsets-idx-start=$subsets_idx_start \
        --subsets-idx-end=$subsets_idx_end | tee \
        ./run_terminal_logs/autoexperiment_subsets_export_te_models/autoexperiment_subsets_export_te_models_output_log_$timestamp.log
else
    # Run without timestamp, explicitly passing "None"
    python3 src/scripts/autoexperiment_subsets_export_te_models.py \
        --timestamp=None \
        --called-from-bash-script=True \
        --subset-len=$subset_len \
        --num-subsets=$num_subsets \
        --subsets-idx-start=$subsets_idx_start \
        --subsets-idx-end=$subsets_idx_end
fi
