#!/bin/bash
# Start the Business Name Generator API

echo "Starting Business Name Generator API..."
echo "Make sure you have installed requirements: pip install -r src/api/requirements.txt"

cd src/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

echo "API started at http://localhost:8000"
echo "Visit http://localhost:8000/docs for interactive API documentation"
