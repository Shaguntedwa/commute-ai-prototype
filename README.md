# 🚍 Commute AI — Smarter, Safer Urban Mobility

**Hackathon: Reimagining Urban Mobility & Daily Commute in India 2026**
Team: shaguntedwa19

## What This Does
Commute AI predicts transit delays and scores route safety
for urban commuters in India — helping them plan smarter,
safer journeys before they leave home.

## Features
- ⏱️ AI Delay Prediction (Random Forest, MAE ~2.3 min)
- 🛡️ Safety Score (transparent weighted formula)
- 👥 Crowd Level Forecast
- 📌 Personalized Recommendations

## Live Demo
👉 [Try it on Hugging Face Spaces](YOUR_HUGGINGFACE_URL_HERE)

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py

## Tech Stack
- Python, Pandas, Scikit-learn
- Gradio (UI)
- Random Forest Regressor (delay model)
- Hugging Face Spaces (deployment)

## Project Structure
commute-ai-prototype/
├── app.py                  # Gradio web app
├── commute_data.csv        # Synthetic training dataset
├── delay_model_predictor.pkl  # Trained ML model
├── requirements.txt        # Dependencies
└── safety_score.ipynb      # Colab notebook (development)
