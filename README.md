# Heart Failure Risk Predictor

Machine learning web app for estimating heart failure risk from clinical parameters. Built as the engineering component of Uzair Saleem's final-year work on cardiovascular disease prediction.

**Live demo:** deploy from this repo, or run locally at `http://localhost:8000`

## What it does

- Compares 6 models: KNN, Decision Tree, Logistic Regression, Naive Bayes, Random Forest, Neural Network
- Selects the best performer using 5-fold cross-validation
- Tunes Random Forest hyperparameters and decision threshold
- Returns risk probability with top 3 explainable factors

## Stack

- Python, FastAPI, scikit-learn
- HTML/CSS frontend served by the API

## Local setup

```bash
cd backend
pip install -r requirements.txt
python train.py
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

## API

- `GET /health` — model status and metrics
- `GET /models` — full model comparison
- `POST /predict` — risk assessment + explainability

## Project structure

```
heart-failure-risk-predictor/
├── backend/          API, training, saved model
├── frontend/         prediction UI
├── data/             heart failure dataset
└── README.md
```
---

Disclaimer: educational use only, not medical advice.
