# Personalized Learning Recommender — Improved Version

This version focuses on understandable recommendations, honest evidence and
beginner-friendly use.

## Main improvements

- Hybrid view combining Course Similarity and User Profile rankings
- Lower and more practical default thresholds
- Transparent fallback when a chosen threshold produces no results
- Raw model score kept visible
- Relative 0–100 match strength clearly labelled as a within-list score
- Rank, confidence label and explanation for every recommended course
- Guidance to select 3–6 representative completed courses
- Adaptive evidence summary for both single-user and multi-user CSV files
- RMSE calculated from an uploaded file only when both actual and predicted
  ratings are present
- Clear strengths and limitations for the recommendation methods
- Beginner guide explaining how to use the app and how predictions are built

## Important honesty note

The relative match strength is not accuracy, probability or RMSE. It is a
normalized display score used only to compare recommendations inside one
generated list.

The Hybrid view is an app-level combination of two completed content-based
methods. It is not presented as a separate trained lab model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload this folder to a GitHub repository and select `app.py` as the entrypoint
in Streamlit Community Cloud.
