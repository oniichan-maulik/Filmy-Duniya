# Filmy Duniya

A movie recommender: a hybrid FastAPI + Streamlit project that lets you scrape TMDB, build a local
in-memory recommendation index, and explore results with a friendly UI.

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Optional: install Playwright browsers if you intend to run UI tests
playwright install

# scrape seed data (or copy your own CSV to data/movies.csv)
python3 -c "from src.scraper import scrape_top_n_movies; scrape_top_n_movies(n=300)"

# start FastAPI backend (recommendation + CSV helpers)
uvicorn src.api:app --reload

# in another shell run the Streamlit UI
streamlit run web/app_streamlit.py
```

### Tests

Backend/unit tests currently consist of the Streamlit UI smoke test.
To run it, start the backend + frontend locally and execute:

```bash
RUN_UI_TESTS=1 pytest
```

By default `pytest` will skip the Playwright test so the suite succeeds
in CI environments where browsers are unavailable.

### Train personalization model

Feedback actions (ratings/favorites) are stored in `data/feedback_events.jsonl`.
Train the personalization ranker from those events:

```bash
python3 -m src.personalization_model --train --min-events 25
python3 -m src.personalization_model --status
```

The model artifact is written to `data/personalization_model.pkl` and is used
by the Streamlit re-ranking flow.

### Mood detection

Mood is detected from the webcam with two pretrained ONNX models in `models/`
(no training needed):

- **Face detection:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  via OpenCV's `FaceDetectorYN`
- **Emotion:** [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition)
  EfficientNet-B2 (7 classes, trained on AffectNet), run with `onnxruntime`

The 7 emotions map to the app's 6 moods (see `EMOTION_TO_MOOD` in
`src/emotion_detection.py`). The HSEmotion weights were trained on AffectNet,
which is licensed for non-commercial research use.

### Import large feedback datasets

If you already have interaction logs in CSV, import them directly into
`data/feedback_events.jsonl`:

```bash
python3 -m src.import_feedback_events --csv /path/to/interactions.csv
```

Useful options:

```bash
# replace existing events file instead of append
python3 -m src.import_feedback_events --csv /path/to/interactions.csv --overwrite

# keep only stronger labels
python3 -m src.import_feedback_events --csv /path/to/interactions.csv --min-abs-signal 0.2
```

### Author

Developed by **Maulik Gupta** ([@oniichan-maulik](https://github.com/oniichan-maulik)).
