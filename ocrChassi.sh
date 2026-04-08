#!/bin/bash
# Launch script for OCR Chassis application

# Change to the application directory
cd "$(dirname "$0")" || exit 1

# Activate the virtual environment and run the application
source venv_linux/bin/activate
python main.py
