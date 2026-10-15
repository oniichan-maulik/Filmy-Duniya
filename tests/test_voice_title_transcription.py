from src.multimodal_mood import _clean_transcript


def test_clean_transcript_removes_common_prefixes():
    assert _clean_transcript("play Inception") == "Inception"
    assert _clean_transcript("recommend me The Matrix") == "The Matrix"
    assert _clean_transcript("movie Interstellar!!!") == "Interstellar"
