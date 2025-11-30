# run-local.ps1
# Helper to run the Streamlit app locally with browser auto-open.
# Activates venv if present, sets STREAMLIT_SERVER_HEADLESS to false, and runs Streamlit.

if (Test-Path .\venv\Scripts\Activate.ps1) {
    Write-Host "Activating virtual environment..."
    & .\venv\Scripts\Activate.ps1
}

# Force Streamlit to open the browser locally
$env:STREAMLIT_SERVER_HEADLESS = 'false'

Write-Host "Starting Streamlit (this will open your browser)..."
streamlit run predictions.py
