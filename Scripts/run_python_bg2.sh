#!/bin/bash

# Check if Python file argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <python_script.py> [session_name]"
    exit 1
fi

# Use provided session name or generate a unique one
SESSION_NAME=${2:-my_python_session_$(date +%s)}

# Create a tmux session and run the Python script inside it
tmux new-session -d -s "$SESSION_NAME" "python $1; exec bash"

echo "Running $1 in a tmux session named '$SESSION_NAME'. Use 'tmux attach -t $SESSION_NAME' to view."

