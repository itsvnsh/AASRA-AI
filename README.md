# AASRA-AI Functional v5 — RAVDESS model

This keeps the v4 interface/modules and replaces the main microphone prediction with a dataset-trained RAVDESS vocal-expression classifier.

## Run
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --reload-exclude '.venv/*'

Open http://127.0.0.1:8000

## Model
- Dataset: RAVDESS 16 kHz speech upload supplied for this project
- Classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Features: 20 MFCC + delta + delta-delta statistics, ZCR, RMS, spectral centroid, spectral rolloff
- Classifier: scaled RBF SVM with probability estimates and class balancing
- Evaluation: actor-disjoint split. Actors 01–18 train (1080 clips), actors 19–24 test (360 clips). Accuracy 51.39%.
- The full confusion/classification report is in model/metrics.txt.

This is a research/hackathon vocal-expression classifier trained on acted English speech. It is not a clinical diagnosis or validated distress assessment and its test accuracy must not be generalized to Indian/multilingual populations.
