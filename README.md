# JUSTICE AI

AI-assisted FIR analysis and investigation support dashboard built for hackathon demos. Upload an FIR image, extract and clean text, generate investigative questions, profiles, roadmap steps, locations, and export/share a concise PDF report.

## Feature Highlights
- Robust OCR pipeline (upscale, denoise, CLAHE, adaptive threshold) for noisy FIR scans.
- AI cleanup of OCR text, with fallback from Gemini to OpenAI when Gemini is restricted.
- Character questions, profiles, locations, and a chronological investigation roadmap generated from FIR context.
- Map visualization of detected locations (MapLibre + Nominatim) and context-aware chat advisor for follow-up questions.
- PDF report generation and optional email delivery via SMTP; rescan flow to iterate quickly after manual edits.

## Requirements
- Python 3.10+ (tested with Flask runtime).
- System dependency: Tesseract OCR must be installed and on PATH.
- pip packages: see `requirements.txt` (Flask, OpenCV, pytesseract, numpy, openai, google-genai, reportlab, etc.).

## Environment Variables
Create a `.env` file in the project root with the keys you need:
- `OPENAI_API_KEY` (required if Gemini is not available)
- `GEMINI_API_KEY` (optional, primary if present)
- `REPORT_EMAIL_USER` / `REPORT_EMAIL_PASSWORD` (optional, needed for email delivery)

## Setup
```bash
python -m venv venv
./venv/Scripts/activate        # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Tesseract (Windows): download from https://github.com/UB-Mannheim/tesseract/wiki and ensure `tesseract.exe` is on PATH.

## Run (dev)
```bash
set FLASK_APP=run.py
flask run
```
Open http://127.0.0.1:5000/ in your browser.

## Usage Flow
1) Upload FIR image → OCR + cleanup runs automatically.
2) Review/edit extracted text and hit Rescan to regenerate AI insights.
3) Explore character cards, roadmap steps, and location map; chat with the FIR-aware advisor.
4) Download PDF or send it via email (requires SMTP credentials in env vars).

## Notes and Troubleshooting
- If OCR is empty or poor, verify Tesseract installation and that the uploaded image is readable; the app already upscales and denoises automatically.
- If AI calls fail, check `GEMINI_API_KEY` or `OPENAI_API_KEY`; the app falls back to OpenAI when Gemini is unavailable.
- For email, Gmail may require app passwords or SMTP access depending on account settings.

## Project Structure
- `run.py` – Flask entrypoint
- `app/routes.py` – routes, AI calls, OCR, PDF/email utilities
- `app/templates/` – HTML templates (dashboard, landing)
- `app/static/` – CSS/JS/assets
- `app/services/` – FIR-specific analysis utilities (e.g., `fir_analyzer.py`)

## License
Internal hackathon project; adapt as needed.
