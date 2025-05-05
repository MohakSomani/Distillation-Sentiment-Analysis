#!/bin/bash

# Set up environment
export TF_CPP_MIN_LOG_LEVEL=3  # Suppress TensorFlow info messages
export PYTHONWARNINGS="ignore"  # Suppress Python warnings

# Create logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Function to run interpretability tools with error handling
run_interpretability() {
    local tool_name=$1
    local script_path=$2
    local output_file=$3
    
    echo "Running $tool_name..."
    
    if python3 "$script_path" > "$output_file" 2>&1; then
        echo "$tool_name completed successfully. Output saved to $output_file"
    else
        echo "Error: $tool_name failed to run. Check $output_file for details."
        return 1
    fi
}

# Run LIME explainer
run_interpretability "LIME" \
    "interpretability/posthoc/lime_explainer.py" \
    "$LOG_DIR/lime_results.html"

# Run SHAP explainer (handled differently since it saves its own output)
echo "Running SHAP..."
if python3 "interpretability/posthoc/shap_explainer.py" 2> "$LOG_DIR/shap_errors.log"; then
    if [ -f "$LOG_DIR/shap_results.html" ]; then
        echo "SHAP completed successfully. Output saved to $LOG_DIR/shap_results.html"
    else
        echo "Error: SHAP ran but didn't produce output. Check $LOG_DIR/shap_errors.log"
    fi
else
    echo "Error: SHAP failed to run. Check $LOG_DIR/shap_errors.log"
fi

# Run other interpretability tools
run_interpretability "Integrated Gradients" \
    "interpretability/self_interpret/feature_importance.py" \
    "$LOG_DIR/feature_importance.log"

run_interpretability "Attention Visualization" \
    "interpretability/self_interpret/attention_viz.py" \
    "$LOG_DIR/attention_viz.log"

echo "All interpretability tools executed."