import json

from src.import_feedback_events import import_feedback_csv


def test_import_feedback_csv_maps_rows_and_skips_invalid(tmp_path):
    csv_path = tmp_path / "events.csv"
    out_path = tmp_path / "feedback_events.jsonl"
    csv_path.write_text(
        "title,genres,director,mood,rating,favorite,media_type\n"
        "Inception,Sci-Fi|Thriller,Christopher Nolan,Happy,5,1,movie\n"
        ",Drama,Someone,sad,1,0,movie\n",
        encoding="utf-8",
    )

    summary = import_feedback_csv(csv_path=csv_path, out_path=out_path, overwrite=True)
    assert summary["rows_total"] == 2
    assert summary["rows_imported"] == 1
    assert summary["rows_skipped_missing_title"] == 1

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["mood"] == "happy"
    assert event["movie"]["title"] == "Inception"
    assert event["movie"]["media_type"] == "movie"
    assert event["favorite"] is True
    assert float(event["signal"]) > 0.0
