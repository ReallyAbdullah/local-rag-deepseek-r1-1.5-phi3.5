#!/bin/bash

echo "Starting the setup process for Local Agentic RAG Application..."
echo "---------------------------------------------------------------"

# --- Configuration ---
PYTHON_COMMAND="python3" # Default to python3
MIN_PYTHON_VERSION="3.9"
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"

# --- Helper Functions ---
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

version_ge() {
    # Compares two version strings (handles versions like 3.10.1 vs 3.9.12)
    # Returns 0 if $1 >= $2, 1 otherwise
    [ "$#" -ne 2 ] && return 1
    # Sort versions and pick the highest. If $2 is highest or they are equal, then $1 >= $2 is false (or true if equal)
    # This logic is a bit tricky with sort; simpler to use Python if available early,
    # but for a shell script, we'll do our best.
    # A more robust way would be:
    #   printf '%s
# %s
# ' "$1" "$2" | sort -V -C
    #   return $?
    # However, sort -V might not be available everywhere.
    # Using a simpler component-wise comparison for now.
    IFS='.' read -r -a v1_parts <<< "$1"
    IFS='.' read -r -a v2_parts <<< "$2"
    
    for i in $(seq 0 $((${#v1_parts[@]} - 1))); do
        if [ -z "${v2_parts[$i]}" ]; then # v1 is longer, so v1 > v2
            return 0
        fi
        if [ "${v1_parts[$i]}" -gt "${v2_parts[$i]}" ]; then
            return 0
        fi
        if [ "${v1_parts[$i]}" -lt "${v2_parts[$i]}" ]; then
            return 1
        fi
    done
    if [ "${#v1_parts[@]}" -lt "${#v2_parts[@]}" ]; then # v2 is longer, e.g. 3.9 vs 3.9.1, so v1 < v2
        return 1
    fi
    return 0 # Versions are equal
}


# --- 1. Check for Python ---
echo ""
echo "Step 1: Checking Python installation..."
if ! command_exists $PYTHON_COMMAND; then
    echo "Error: $PYTHON_COMMAND is not installed or not in PATH."
    echo "Please install Python $MIN_PYTHON_VERSION or later and try again."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_COMMAND -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Found Python version: $PYTHON_VERSION"

if ! version_ge "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
    echo "Error: Python version $PYTHON_VERSION is less than the required $MIN_PYTHON_VERSION."
    echo "Please upgrade your Python installation and try again."
    exit 1
fi
echo "Python version check passed."

# --- 2. Create Virtual Environment ---
echo ""
echo "Step 2: Setting up Python virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
    read -p "Do you want to remove and recreate it? (y/N): " recreate_venv
    if [[ "$recreate_venv" =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        echo "Creating new virtual environment in '$VENV_DIR'..."
        $PYTHON_COMMAND -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create virtual environment."
            exit 1
        fi
    else
        echo "Skipping virtual environment creation. Using existing one."
    fi
else
    echo "Creating new virtual environment in '$VENV_DIR'..."
    $PYTHON_COMMAND -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi
echo "Virtual environment setup complete."

# --- 3. Activate Virtual Environment and Install Dependencies ---
echo ""
echo "Step 3: Installing dependencies..."
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    echo "Please try activating it manually: source $VENV_DIR/bin/activate"
    exit 1
fi
echo "Virtual environment activated."

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found."
    echo "Please ensure you are in the root directory of the project and the file exists."
    exit 1
fi

echo "Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    echo "Please check the output above for errors. You might need to install some system libraries manually."
    exit 1
fi
echo "Dependencies installed successfully."

# --- 4. Reminders for Manual Steps ---
echo ""
echo "---------------------------------------------------------------"
echo "Step 4: Manual Prerequisites - IMPORTANT!"
echo "---------------------------------------------------------------"
echo "Please ensure you have the following installed and configured:"
echo ""
echo "1. Ollama:"
echo "   - Install from: https://ollama.ai/"
echo "   - Ensure the Ollama application is running before starting the app."
echo ""
echo "2. Ollama Models (run these commands in your terminal):"
echo "   ollama pull nomic-embed-text"
echo "   ollama pull phi3.5"
echo "   ollama pull deepseek-r1:1.5b"
echo ""
echo "3. Tesseract OCR:"
echo "   - For macOS: brew install tesseract"
echo "   - For Ubuntu/Debian: sudo apt install tesseract-ocr"
echo "   - For other Linux distributions, please consult your package manager."
echo "   - Ensure Tesseract is in your system's PATH."
echo "---------------------------------------------------------------"

# --- 5. How to Run ---
echo ""
echo "Setup complete!"
echo "To run the application:"
echo "1. Ensure Ollama is running."
echo "2. Activate the virtual environment (if not already active in this session):"
echo "   source $VENV_DIR/bin/activate"
echo "3. Run the application:"
echo "   python app.py"
echo ""
echo "Access the UI at http://localhost:7860 (or the configured port)."
echo "---------------------------------------------------------------"

exit 0
