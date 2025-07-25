#!/bin/bash

# Exit on error
set -e

# 1. Create and activate virtual environment
if [ ! -d "backend/.venv" ]; then
  python -m venv backend/.venv
fi
source backend/.venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up the database
python scripts/setup_db.py

# 4. Start the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
