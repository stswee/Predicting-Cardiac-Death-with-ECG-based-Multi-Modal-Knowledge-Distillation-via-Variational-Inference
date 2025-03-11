#!/bin/bash

# Check if two Python file arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <python_script1.py> <python_script2.py>"
    exit 1
fi

# Define session names
SESSION1="session1"
SESSION2="session2"

# Start tmux sessions and run scripts
tmux new-session -d -s $SESSION1 "python $1; exec bash"
tmux new-session -d -s $SESSION2 "python $2; exec bash"

echo "Running $1 in a tmux session named '$SESSION1'."
echo "Running $2 in a tmux session named '$SESSION2'."
echo "Use 'tmux attach -t session1' or 'tmux attach -t session2' to view."

