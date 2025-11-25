# AI-powered Resume Screening and Ranking System

A modular system that automates resume parsing, candidate scoring, and ranking using machine learning and NLP. The project includes data preprocessing, resume parsing, feature extraction, a ranking model, and a web UI for recruiters.

## Key features
- Resume parsing (PDF/DOCX) → structured fields (name, education, skills, experience)
- Skill/keyword extraction and normalization
- Feature engineering: years of experience, role match score, skill overlap, education level
- Candidate scoring & ranking model (ML or hybrid rules + model)
- Batch and single-resume evaluation modes
- Web UI for recruiter workflows (upload, view ranked list, export)
- API endpoints for integration with ATS

## Tech stack
- Python (pandas, scikit-learn, transformers/spacy)
- Resume parsing: `pdfminer` / `python-docx` / `pyresparser` (or custom)
- Web UI: FastAPI + optional React or Streamlit/Gradio for demo
- Persistence: PostgreSQL / SQLite (for demo)
- Containerization: Docker

## Quick start (dev)
1. Unzip the project and enter the folder.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows
   pip install -r requirements.txt
