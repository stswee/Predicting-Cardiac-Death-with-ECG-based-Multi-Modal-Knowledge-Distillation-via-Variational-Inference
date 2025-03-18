#!/bin/bash

# Check if Python file argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

# Create a tmux session and run the Python script inside it
tmux new-session -d -s my_python_session2 "python $1; exec bash"
echo "Running $1 in a tmux session named 'my_python_session2'. Use 'tmux attach -t my_python_session' to view."
