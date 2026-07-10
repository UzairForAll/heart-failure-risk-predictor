---
title: QalbRisk
emoji: ❤️
colorFrom: teal
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# QalbRisk · Heart Failure Assessment

Machine learning web app for estimating heart failure risk from clinical parameters.

**Built by [Uzair Saleem](https://hellouzair.com)**

## What it does

- Compares 6 models and uses the best performer (Random Forest)
- 5-fold cross-validation, hyperparameter tuning, threshold tuning
- Returns risk probability with top 3 explainable factors

## Local setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

## API

- `GET /health` — model status and metrics
- `GET /models` — full model comparison
- `POST /predict` — risk assessment + explainability

Disclaimer: educational use only, not medical advice.
