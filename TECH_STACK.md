# Tech Stack & Algorithms — ReviewerAI

This document summarizes the technologies, libraries, and core techniques used to build ReviewerAI — an AI-driven research paper reviewer that analyzes uploaded papers (PDF/DOCX), performs plagiarism and citation analysis, generates rewrite suggestions, and provides interactive visualizations.

## Overview

- ReviewerAI is a Flask-based web application with a server-rendered UI (Jinja2) and a rich JavaScript frontend for interactive visualizations and user workflows. It integrates LLMs (OpenAI / Gemini) for analysis, rewrites, and question generation and uses rule-based and statistical algorithms for plagiarism, formatting, and citation checks.

## Main Technologies

- **Python 3.x** — Backend logic, analysis pipelines, LLM orchestration
- **Flask** — Web framework (routes, API endpoints, templating)
- **Jinja2** — Server-side HTML templates
- **JavaScript (ES6+)** — Frontend interactivity, canvas/SVG visualizations
- **HTML5 / CSS3** — Responsive UI and visual theming
- **SQLite** — Local persistence for lightweight data (optional)
- **dotenv** — Environment configuration for API keys and secrets

## Key Libraries & Services

- **OpenAI API** — Primary LLM integration for text analysis, rewrites, chat, and TTS
- **Google Gemini (optional)** — Alternative LLM provider with fallback logic
- **pdfplumber** — Precise PDF text extraction (page/position aware)
- **python-docx** — DOCX extraction support
- **OpenCV (cv2)** — Image decoding and optional image utilities
- **reportlab** — Server-side PDF generation for downloadable reports
- **html2pdf.js (html2canvas + jsPDF)** — Client-side PDF export of results page
- **Marked.js + DOMPurify** — Safe rendering of markdown content in chat UI
- **Spline Viewer** — Decorative 3D background component (via CDN)

## Frontend Visualizations & UX

- **Source Type Distribution** — Interactive SVG pie chart with hover/legend filtering
- **Reference Constellation** — Canvas-based force/oscillation layout showing reference relationships; supports hover, zoom, and info panel
- **Radar Chart** — SVG polygon plot for quality metrics (style, originality, vocabulary, etc.) with cursor sweep
- **Rewrite / Opportunities UI** — Cards and modals to generate rewrites (section or targeted), compare originals, and view suggested changes

## Core Algorithms & Techniques

### Plagiarism & Similarity

- Text similarity: Jaccard overlap, n-gram / shingle comparisons, and sequence matching
- MinHash / shingling approach for scalable similarity (sketching) where applicable
- Web-source checks (DuckDuckGo / Google Scholar) to find potential matches and academic sources
- Sentence-level scoring and extraction to surface highest-risk passages
- Paraphrase detection heuristics via pattern analysis and lexical variance metrics

### Citation & Reference Analysis

- Reference parsing: heuristics and lightweight parsers to extract structured citation fields
- Reference verification: search / metadata checks and confidence scoring
- Citation density and unsupported claim detection (flag claims lacking citations)

### Formatting & Document Quality

- Rule-based checks (abstract presence, references section, heading consistency)
- AI-assisted formatting suggestions using LLM completions when available
- Page-level scoring and aggregate document grade

### Rewrite / Improvement Engine

- LLM-driven rewrites for sections or targeted sentences, with prompts engineered to return JSON-like structured changes
- Opportunity detection: heuristic + model-aided scoring to prioritize high-impact rewrites

## Extraction & Preprocessing

- PDF parsing with positional awareness (`pdfplumber`) to map citations/phrases to pages and coordinates
- DOCX parsing with `python-docx`
- Text cleaning and normalization (whitespace, unicode, punctuation normalization)

## Reporting & Export

- Server-side PDF generation using `reportlab` for official report exports
- Client-side PDF export (html2pdf.js) for quick downloadable snapshots of the results page

## Environment & Configuration

- Store secrets in a `.env` file (API keys, email credentials, optional Gemini Apps Script URL)
- Main env variables used: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GEMINI_APPS_SCRIPT_URL`, `REPORT_EMAIL_USER`, `REPORT_EMAIL_PASSWORD`, `SECRET_KEY`

## Extensibility & Optional Modules

- The project includes optional modules that were present in the original FIR codebase and can be enabled as needed (e.g., face-matching utilities using DeepFace/OpenCV). These are not required for core ReviewerAI functionality but can be repurposed for auxiliary features.

## Deployment Notes

- Install dependencies from `requirements.txt` and set environment variables via `.env`.
- Run locally with `flask run` for development; for production, use a WSGI server (Gunicorn / uWSGI) behind a reverse proxy.

---
This file describes the ReviewerAI architecture and algorithms. For implementation details, consult the associated modules in `app/` (routes, services, templates) and `requirements.txt` for exact dependency versions.
