# Movie Recommender Task: Remove Modes & Set Refresh Rates + Train Models

## Step 1: Create/Update TODO.md [COMPLETED]

## Step 2: Edit web/app_streamlit.py
- Remove `continuous_voice = st.sidebar.checkbox("🎤 Continuous Voice Mode", value=False)`
- Set `continuous_voice = False` hardcoded
- Remove `AUTO_REFRESH = st.sidebar.checkbox("Enable auto-refresh (poll backend)", value=False)`
- Set `AUTO_REFRESH = False` hardcoded  
- Hardcode `scan_interval_sec = 6.0` in mood scanning loop (remove continuous_voice condition)
- Remove live_voice.html component block (if continuous_voice)
- Disable auto-refresh sleep loop

## Step 3: Test UI Changes
- Run `streamlit run web/app_streamlit.py`
- Verify: No checkboxes appear in sidebar
- Verify: Mood scans every ~6s (check captions/timing)
- Select movie/watch: Scanning pauses (manual_override/watch page)
- Voice duration remains 2s

## Step 4: Train Personalization Model
- Check events: `python -m src.personalization_model --status`
- Train: `python -m src.personalization_model --train --min-events 5`

## Step 5: Train Emotion Model  
- Ensure FER data: `python -m src.import_fer_dataset`
- Train: `python -m src.train_emotion_model --train-dir data/fer2013/train --test-dir data/fer2013/test`

## Step 6: Final Verification
- Restart app, test full flow: mood→recommend→select movie (pauses refresh)
- Check model status: `python -m src.personalization_model --status`
- Update this TODO with completion marks

**Next step: Edit web/app_streamlit.py**
