# AI-Powered Resume Screening System

> H3S Model — Hybrid Semantic-Structural Scoring for automated, 
> bias-reduced candidate ranking.  
> **Published at IEEE ICCCA 2025** → https://ieeexplore.ieee.org/document/11325535

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![RoBERTa](https://img.shields.io/badge/RoBERTa-Embeddings-orange?style=flat-square)
![NER](https://img.shields.io/badge/NLP-Named_Entity_Recognition-green?style=flat-square)
![F1](https://img.shields.io/badge/F1_Score-88.6%25-brightgreen?style=flat-square)

## What it does

An end-to-end ML pipeline that parses resumes, extracts structured candidate 
information, and ranks applicants using a hybrid scoring model that combines 
semantic understanding with structural analysis — going far beyond simple 
keyword matching.

## How the H3S Model works
Resume (PDF/DOCX)
│
▼
┌───────────────────┐
│   Resume Parser   │  ← Extracts raw text
└────────┬──────────┘
│
▼
┌───────────────────┐     ┌──────────────────────┐
│  RoBERTa Encoder  │     │   NER Extractor      │
│  (Semantic Score) │     │  skills, experience, │
│                   │     │  education, titles   │
└────────┬──────────┘     └──────────┬───────────┘
│                           │
└──────────┬────────────────┘
▼
┌─────────────────────┐
│   H3S Scoring Model │
│  Hybrid weighted    │
│  ranking output     │
└─────────────────────┘
│
▼
Ranked candidate list

## Results

| Metric | Score |
|---|---|
| F1-Score | **88.6%** |
| Improvement over keyword baseline | **+14 percentage points** |
| Test set size | 500 resumes |

## Tech Stack

- **Language:** Python 3.9+
- **Embeddings:** RoBERTa (sentence-transformers)
- **NLP:** spaCy NER, custom entity extraction
- **ML:** Scikit-learn, PyTorch
- **API:** Flask REST API
- **Parsing:** PDFMiner, python-docx

## Setup & Run
```bash
# 1. Clone the repo
git clone https://github.com/vikasmishra16/ai-resume-screener.git
cd ai-resume-screener

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Run the Flask API
python app.py

# Or explore the notebook
jupyter notebook
```

## Project Structure
ai-resume-screener/
├── app.py                  # Flask API entry point
├── model/                  # H3S model code
├── parser/                 # Resume parsing utilities
├── data/                   # Sample resumes and datasets
├── notebooks/              # Jupyter exploration notebooks
├── requirements.txt        # Python dependencies
└── README.md

## Research Publication

This project is the basis of a peer-reviewed IEEE paper:

**"A Hybrid Semantic-Structural Approach for Fair and Accurate 
Automated Resume Screening"**  
IEEE ICCCA 2025 — https://ieeexplore.ieee.org/document/11325535

## Author

Built by [Vikas Mishra](https://github.com/vikasmishra16)  
Part of research published at IEEE ICCCA 2025
