# Start the Business Name Generator API on Windows

Write-Host "Starting Business Name Generator API..."
Write-Host "Make sure you have installed requirements: pip install -r src/api/requirements.txt"

Set-Location src/api
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Write-Host "API started at http://localhost:8000"
Write-Host "Visit http://localhost:8000/docs for interactive API documentation"
