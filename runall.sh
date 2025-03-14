#!/bin/bash
# Adjust the paths to your Python scripts as needed.

# Get the absolute path to the directory containing the scripts.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Open a new Terminal window and run sph.py, position at (0,0)
osascript <<EOF
tell application "Terminal"
    do script "cd '$SCRIPT_DIR'; python3 sph.py"
    delay 0.5
    set bounds of front window to {0, 0, 300, 300}
end tell
EOF

# Open a new Terminal window and run sph_cython.py, position at (810,0)
osascript <<EOF
tell application "Terminal"
    do script "cd '$SCRIPT_DIR'; python3 sph_cython.py"
    delay 0.5
    set bounds of front window to {300, 0, 600, 300}
end tell
EOF

# Open a new Terminal window and run sph_serial_dask.py, position at (0,610)
osascript <<EOF
tell application "Terminal"
    do script "cd '$SCRIPT_DIR'; python3.12 sph_serial_dask.py"
    delay 0.5
    set bounds of front window to {600, 0, 900, 300}
end tell
EOF
