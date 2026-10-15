# Emotion-Aware Movie Recommender System: A Research Paper

## Abstract

This document provides a comprehensive technical explanation of the Emotion-Aware Movie Recommender System, covering the architecture, design decisions, algorithms, and implementation details. The system combines content-based filtering with real-time multimodal emotion detection to provide personalized movie recommendations based on user facial expressions and voice analysis.

---

## 1. Introduction

### 1.1 Problem Statement

Traditional movie recommendation systems rely on collaborative filtering or content-based approaches that analyze movie metadata and user viewing history. However, these systems fail to capture the user's emotional state, which is a critical factor in determining movie preferences. A user in a happy mood may prefer light-hearted comedies, while someone feeling anxious might enjoy calming documentaries.

### 1.2 Proposed Solution

We propose an emotion-aware movie recommender system that:

1. **Detects user mood in real-time** using facial expression recognition and voice analysis
2. **Maps emotions to genres** using a carefully designed mood-to-genre mapping
3. **Provides personalized recommendations** using content-based filtering with TF-IDF or sentence embeddings
4. **Learns from user feedback** to continuously improve recommendation quality

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EMOTION-AWARE MOVIE RECOMMENDER                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Web UI     │    │  REST API    │    │   Scraper    │                  │
│  │ (Streamlit)  │    │  (FastAPI)   │    │   (TMDB)     │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             │                                                │
│                    ┌────────▼────────┐                                       │
│                    │   Core Engine   │                                       │
│                    │  (Recommender)   │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│         ┌───────────────────┼───────────────────┐                           │
│         │                   │                   │                           │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐                 │
│  │   Emotion    │    │  Multimodal  │    │Personality   │                 │
│  │  Detection   │    │    Mood      │    │   Model      │                 │
│  │   (Face)      │    │  (Face+Voice)│    │  (Learning)  │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                    │   │
│  │  movies.csv │ feedback_events.jsonl │ user_learning.json │ cache/   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Description

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web UI | Streamlit | User interface for movie browsing and feedback |
| REST API | FastAPI | Backend services for recommendations |
| Scraper | TMDB API | Automated movie data collection |
| Emotion Detection | OpenCV + TensorFlow | Facial expression recognition |
| Multimodal Mood | Audio + Video fusion | Combined face and voice analysis |
| Personalization | Scikit-learn | Learning from user feedback |
| Vectorization | TF-IDF / SBERT | Text feature extraction |

---

## 3. Data Collection and Management

### 3.1 Movie Data Source

The system uses **The Movie Database (TMDB)** API as the primary data source. TMDB provides:

- Movie titles, overviews, and synopses
- Genre information
- Cast and crew details
- Director information
- User ratings and popularity scores
- Poster images

### 3.2 Scraper Module (`src/scraper.py`)

**Design Decision: Why TMDB?**
- Free tier available with reasonable API limits
- Comprehensive metadata including cast/director
- Clean REST API with JSON responses
- Supports both movies and TV shows

**Key Design Decisions:**

1. **Rate Limiting**: API calls are throttled to 0.2-0.25 seconds between requests to respect TMDB's rate limits.

2. **Deduplication**: The scraper checks existing movies by both `movie_id` and `title` to avoid duplicates.

3. **Batch Processing**: Supports incremental updates via `max_per_run` parameter to prevent overwhelming the API.

4. **Error Handling**: Failed API calls are logged but don't stop the entire scraping process.

```python
# Scraper flow:
fetch_popular() → fetch_details() → _normalize_detail() → append_movie()
```

### 3.3 Data Storage

**CSV Format** (`data/movies.csv`):
```
movie_id,media_type,title,genres,cast,director,description,rating,popularity,poster_path
movie:550,Action|Thriller,Die Hard,Bruce Willis|Alan Rickman,John McTiernan,1988 Christmas heist...,8.3,120.5,/path/to/poster.jpg
```

**Design Decision: Why CSV instead of Database?**
- Simplicity for single-user/local deployment
- Easy to version control with Git
- No external database dependency
- Sufficient for datasets under 100K movies

---

## 4. Content-Based Filtering

### 4.1 Feature Engineering (`src/preprocess.py`)

The system creates a "soup" of movie features by combining:

- **Title**: Movie name
- **Genres**: Genre categories (Action, Comedy, Drama, etc.)
- **Director**: Director's name
- **Cast**: Top cast members (up to 10)
- **Description**: Plot overview

**Design Decision: Why Combine All Text Fields?**
- TF-IDF can identify important terms across all fields
- Creates a rich representation that captures multiple aspects
- Simple yet effective approach without complex feature engineering

### 4.2 Vectorization Methods

The system supports two vectorization approaches:

#### 4.2.1 TF-IDF (Default)

**Algorithm: Term Frequency-Inverse Document Frequency**

```
TF(t,d) = (Number of times term t appears in document d) / (Total terms in d)
IDF(t) = log(Total documents / Documents containing term t)
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Configuration**:
```python
TfidfVectorizer(
    stop_words='english',  # Remove common words
    max_features=20000       # Limit vocabulary size
)
```

**Design Decision: Why TF-IDF?**
- Well-understood, interpretable
- Fast computation
- Good baseline performance
- Works well with limited data

#### 4.2.2 Sentence Embeddings (Optional)

**Algorithm**: Sentence-BERT (SBERT)

```python
# Uses "all-MiniLM-L6-v2" model
# 384-dimensional dense embeddings
# Captures semantic meaning better than TF-IDF
```

**Design Decision: Why SBERT?**
- Pre-trained on massive corpora
- Captures semantic similarity (not just lexical)
- Fast inference with the small model variant
- Toggle via `USE_EMBEDDINGS` environment variable

### 4.3 Similarity Computation

**Algorithm**: Cosine Similarity

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Implementation**:
```python
from sklearn.metrics.pairwise import cosine_similarity

# For N movies with M features:
# similarity_matrix[i][j] = similarity between movie i and movie j
sim = cosine_similarity(tfidf_matrix)
```

**Design Decision: Why Cosine Similarity?**
- Works well with high-dimensional sparse vectors
- Magnitude-agnostic (focuses on direction)
- Efficient to compute for large matrices
- Well-suited for text similarity

---

## 5. Recommendation Algorithms

### 5.1 Recommendation by Title (`recommend_by_title`)

**Algorithm Flow:**

1. **Input**: Movie title string
2. **Search**: Find exact match, fallback to fuzzy matching using `difflib.get_close_matches`
3. **Similarity**: Get similarity scores for the input movie
4. **Ranking**: Sort by similarity score (descending)
5. **Output**: Top-N similar movies (excluding the input movie itself)

**Design Decision: Why Fuzzy Matching?**
- Handles typos and minor title variations
- Improves user experience significantly
- Threshold of 0.6 provides good balance

### 5.2 Recommendation by Genre (`recommend_by_genre`)

**Algorithm Flow:**

1. **Input**: Genre string
2. **Filtering**: Find all movies containing the genre
3. **Centroid Computation**: Calculate the mean vector of all movies in that genre
4. **Ranking**: Sort genre movies by similarity to centroid

**Mathematical Formulation**:

```python
# Centroid of genre G:
centroid_G = mean(v1, v2, ..., vk)  # where vi are genre movie vectors

# Similarity score for movie m in genre G:
score(m, G) = cosine_similarity(vector(m), centroid_G)
```

**Design Decision: Why Centroid Approach?**
- Captures the "essence" of a genre
- Allows ranking within a genre
- More nuanced than simple filtering

### 5.3 Recommendation by Mood (`recommend_by_mood`)

**Mood-to-Genre Mapping**:

| Mood | Recommended Genres |
|------|-------------------|
| Happy | Comedy, Romance, Family |
| Sad | Drama, Feel-Good |
| Angry | Action, Thriller |
| Anxious | Animation, Comedy |
| Excited | Adventure, Sci-Fi |
| Calm | Documentary, Slice of Life |

**Algorithm Flow:**

1. **Input**: Mood string (e.g., "happy", "sad")
2. **Genre Mapping**: Look up genres for the mood
3. **Filtering**: Find movies matching any of the mapped genres
4. **Ranking**: Use centroid-based similarity within filtered set

**Design Decision: Why This Mapping?**
- Based on psychological research on mood regulation
- Comedy/Romance for happiness (positive emotions)
- Drama/Feel-Good for sadness (comfort movies)
- Action/Thriller for anger (high-arousal)
- Documentary/Calm for anxiety (low-arousal, soothing)

---

## 6. Emotion Detection System

### 6.1 Facial Emotion Recognition (`src/emotion_detection.py`)

**Model Architecture**: Mini-XCEPTION

- **Input**: 48×48 grayscale face images
- **Output**: 7 emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Framework**: TensorFlow/Keras
- **Pre-training**: FER2013 dataset

**Processing Pipeline:**

```
Frame Capture (OpenCV) 
    → Face Detection (Haar Cascade)
    → Face Extraction & Preprocessing
    → Model Inference
    → Emotion Classification
    → Mood Mapping
```

**Design Decisions:**

1. **Why Haar Cascade for Face Detection?**
   - Lightweight and fast
   - Works in real-time on CPU
   - No deep learning overhead

2. **Why CLAHE Preprocessing?**
   - Enhances contrast for better feature extraction
   - Improves robustness to lighting variations
   - Standard technique in FER systems

3. **Why Emotion-to-Mood Mapping?**
   - Direct emotion-to-genre mapping is too granular
   - Mood provides a better abstraction level
   - Reduces the number of genre categories needed

### 6.2 Emotion-to-Mood Mapping

```python
EMOTION_TO_MOOD = {
    "Happy": "happy",
    "Sad": "sad",
    "Angry": "angry",
    "Disgust": "angry",    # Similar arousal level
    "Fear": "anxious",
    "Surprise": "excited",
    "Neutral": "calm",
}
```

**Weighted Mapping for Robustness**:

```python
_MOOD_WEIGHTS = {
    "Happy": {"happy": 0.9, "excited": 0.25},
    "Sad": {"sad": 1.0, "calm": 0.1},
    # ... etc
}
```

**Design Decision: Why Weighted Approach?**
- Emotions aren't binary
- Multiple moods can apply simultaneously
- Provides smoother transitions between states

---

## 7. Multimodal Mood Detection

### 7.1 Voice Analysis (`src/multimodal_mood.py`)

**Algorithm**: Rule-based audio feature analysis

**Audio Features Extracted:**

1. **RMS (Root Mean Square)**: Measures loudness/energy
   - High RMS → excited, angry, happy
   - Low RMS → sad, calm

2. **Zero Crossing Rate (ZCR)**: Measures signal complexity
   - High ZCR → anxious, excited
   - Low ZCR → calm, sad

3. **Peak Amplitude**: Maximum signal magnitude
   - High peaks → angry, excited

**Mood Scoring Rules:**

```python
scores["excited"] += min(1.0, rms * 7.0) + min(0.8, zcr * 12.0)
scores["angry"] += min(1.0, rms * 8.5) + min(1.0, peak * 0.8)
scores["happy"] += min(1.0, rms * 6.0) + min(0.5, zcr * 6.0)
# ... etc
```

**Design Decision: Why Rule-Based?**
- No training data needed for voice
- Interpretable and adjustable
- Fast inference
- Complementary to visual emotion detection

### 7.2 Signal Fusion

**Algorithm**: Weighted Score Fusion

```
fused_score(mood) = face_weight × face_score(mood) + voice_weight × voice_score(mood)
```

**Default Weights**:
- Face: 70%
- Voice: 30%

**Design Decision: Why Face-Heavy?**
- Facial expressions are more reliable indicators
- Voice can be affected by recording quality
- Face provides continuous monitoring

### 7.3 Speech-to-Text (Optional)

**Purpose**: Allow users to speak movie titles

**Implementation**: Google Speech Recognition API

**Fallback**: PocketSphinx for offline operation

---

## 8. Personalization System

### 8.1 Feedback Collection (`src/user_store.py`)

**Feedback Types:**

1. **Explicit Ratings**: 1-5 star ratings
2. **Implicit Signals**: Liking a recommendation (+0.6)
3. **Mood Context**: What mood the user was in

**Feedback Event Schema:**

```json
{
    "timestamp": "2024-01-15T10:30:00Z",
    "mood": "happy",
    "rating": 4.5,
    "favorite": true,
    "signal": 1.0,
    "movie": {
        "title": "The Grand Budapest Hotel",
        "genres": "Comedy|Adventure",
        "director": "Wes Anderson"
    }
}
```

**Design Decision: Why JSONL Format?**
- Append-only (good for logging)
- Easy to parse line-by-line
- Can be processed incrementally
- Human-readable for debugging

### 8.2 Personalization Model (`src/personalization_model.py`)

**Algorithm**: SGD Regressor with Huber Loss

**Feature Engineering:**

```python
def _feature_dict(movie, mood):
    features = {}
    
    # Mood context
    if mood:
        features[f"mood={mood}"] = 1.0
    
    # Media type
    features[f"media_type={movie.media_type}"] = 1.0
    
    # Genres (one-hot)
    for genre in movie.genres:
        features[f"genre={genre}"] = 1.0
    
    # Director
    features[f"director={movie.director}"] = 1.0
    
    # Numerical features
    features["rating"] = movie.rating
    features["popularity"] = movie.popularity
    
    return features
```

**Model Configuration:**

```python
SGDRegressor(
    loss="huber",        # Robust to outliers
    alpha=1e-4,          # L2 regularization
    penalty="l2",
    random_state=42,
    max_iter=3000,
    tol=1e-3
)
```

**Design Decision: Why Huber Loss?**
- Combines MSE and MAE
- Less sensitive to outliers than MSE
- Provides smooth gradients

### 8.3 Recommendation Re-ranking

**Algorithm**: Score Boosting

```python
def rerank_results_for_learning(results, mood):
    for movie in results:
        # Rule-based boost
        rule_boost = _learning_score(movie, mood, profile)
        
        # ML model boost
        model_boost = predict_personalization_boost(movie, mood)
        
        # Combined boost with clipping
        total_boost = clip(rule_boost + model_boost, -1.2, 1.2)
        
        movie.score += total_boost
    
    return sorted(results, key=lambda x: x.score, reverse=True)
```

---

## 9. REST API Design

### 9.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommend` | GET | Get recommendations by title or genre |
| `/add_movie` | POST | Add new movie to database |
| `/refresh` | POST | Refresh recommendation index |
| `/personalization/status` | GET | Get model training status |
| `/personalization/train` | POST | Trigger model training |

### 9.2 Background Tasks

**Periodic Refresh** (every 24 hours by default):
1. Scrape new movies from TMDB
2. Rebuild recommendation index
3. Update caches

**Periodic Training** (every 5 minutes):
1. Check for new feedback events
2. Retrain personalization model if needed

---

## 10. User Interface (Streamlit)

### 10.1 UI Components

1. **Movie Browser**: Search and filter movies
2. **Mood Detection Panel**: Webcam-based emotion detection
3. **Recommendation Cards**: Display movies with posters
4. **Feedback Buttons**: Like/Dislike and rating
5. **Video Player**: Stream movies (optional)

### 10.2 State Management

- Session state for user preferences
- Cached movie data
- Real-time mood detection

---

## 11. Performance Considerations

### 11.1 Caching Strategy

| Cache | Location | Purpose |
|-------|----------|---------|
| TF-IDF Matrix | `data/cache/tfidf_matrix.npy` | Vector representations |
| TF-IDF Vectorizer | `data/cache/tfidf_vectorizer.pkl` | Vocabulary |
| Similarity Matrix | `data/cache/tfidf_sim.npy` | Pre-computed similarities |
| Embeddings | `data/cache/embeddings.npy` | SBERT vectors |

### 11.2 Lazy Loading

- Movie data loaded on-demand
- Emotion model loaded once and cached
- Index built at startup or on-refresh

### 11.3 Thread Safety

- File-level locks for CSV operations
- Thread-safe index updates
- Concurrent API request handling

---

## 12. Configuration System

### 12.1 Environment Variables

```python
TMDB_API_KEY = "785c5f1bd5e3e823f06abdfe6168588e"  # Default (limited)
USE_EMBEDDINGS = "0"  # Set to "1" for SBERT
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
REFRESH_INTERVAL_SECONDS = 86400  # Daily
SCRAPE_MOVIE_COUNT = 400
SCRAPER_MAX_PER_RUN = 100
ENABLE_AUTO_SCRAPER = "1"
```

---

## 13. Testing Strategy

### 13.1 Test Types

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: API endpoint testing
3. **UI Tests**: Playwright-based browser testing

### 13.2 Test Coverage

- `test_personalization_model.py`: Model training and prediction
- `test_import_feedback_events.py`: Feedback parsing
- `test_playwright_ui.py`: End-to-end UI testing

---

## 14. Deployment

### 14.1 Docker Support

The system includes Docker Compose configuration for:

- Backend API service
- Frontend Streamlit app
- Scraper service

### 14.2 Systemd Services (Optional)

For production Linux deployments:

- `movie_recommender-backend.service`
- `movie_recommender-frontend.service`
- `movie_recommender-scraper.service`

---

## 15. Conclusion

This emotion-aware movie recommender system demonstrates how multimodal情感 detection can enhance traditional recommendation systems. By combining:

1. **Content-based filtering** with TF-IDF/SBERT
2. **Real-time emotion detection** from facial expressions
3. **Voice analysis** for multimodal mood detection
4. **Personalization** through feedback learning

The system provides a more intuitive and personalized user experience that adapts to the user's emotional state.

### 15.1 Future Work

1. **Collaborative Filtering**: Add user-based or item-based CF
2. **Deep Learning**: Use neural networks for better embeddings
3. **Context Awareness**: Consider time of day, day of week
4. **Social Features**: Friends' recommendations
5. **Streaming Integration**: Direct streaming links

---

## Appendix: File Structure

```
movie_recommender/
├── src/
│   ├── api.py                    # FastAPI backend
│   ├── config.py                  # Configuration
│   ├── data__loader.py            # CSV operations
│   ├── emotion_detection.py       # Face-based emotion
│   ├── multimodalmood.py          # Voice + face fusion
│   ├── personalization_model.py   # ML personalization
│   ├── preprocess.py              # Feature engineering
│   ├── recomender.py              # Core algorithms
│   ├── scraper.py                 # TMDB scraper
│   ├── user_store.py              # Feedback storage
│   └── vectorize.py               # TF-IDF/SBERT
├── web/
│   └── app_streamlit.py           # Streamlit UI
├── data/
│   ├── movies.csv                 # Movie database
│   ├── feedback_events.jsonl      # User feedback
│   ├── user_learning.json         # Learning profile
│   ├── personalization_model.pkl # Trained model
│   └── cache/                     # Pre-computed features
├── tests/                         # Test suite
├── docker-compose.yml             # Docker config
└── RESEARCH_PAPER.md              # This document
```

---

*Document Version: 1.0*
*Last Updated: 2024*
