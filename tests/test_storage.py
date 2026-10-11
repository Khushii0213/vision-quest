import json

import pytest

from src.storage import MAX_RUNS, Storage, StorageError


def summary(score=300, correct=3, wrong=1, streak=2, rt=10.0, answered=4):
    return {"difficulty": "normal", "score": score, "correct": correct, "wrong": wrong, "timeouts": 0,
            "skipped": 0, "rounds_played": answered, "answered": answered, "response_time_total": rt,
            "avg_response_s": rt / answered, "best_streak": streak, "end_reason": "completed"}


def test_missing_file_gives_defaults(tmp_path):
    d = Storage(tmp_path / "s.json").load()
    assert d["best_score"] == 0 and d["runs"] == []


def test_record_and_persist_across_instances(tmp_path):
    p = tmp_path / "sub" / "s.json"
    info = Storage(p).record_run(summary(300))
    assert info == {"new_best": True, "previous_best": 0}
    info2 = Storage(p).record_run(summary(200, streak=5))
    assert info2["new_best"] is False
    d = Storage(p).load()
    assert d["best_score"] == 300 and d["best_streak"] == 5 and len(d["runs"]) == 2
    assert d["rounds_played"] == 8 and Storage.average_response(d) == 2.5


def test_corrupt_file_is_preserved_and_replaced(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert Storage(p).load()["best_score"] == 0
    assert (tmp_path / "s.corrupt").exists()


def test_wrong_structure_treated_as_corrupt(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([1, 2]))
    assert Storage(p).load()["runs"] == []


def test_history_capped(tmp_path):
    s = Storage(tmp_path / "s.json")
    for i in range(MAX_RUNS + 5):
        s.record_run(summary(i))
    assert len(s.load()["runs"]) == MAX_RUNS


def test_reset(tmp_path):
    s = Storage(tmp_path / "s.json")
    s.record_run(summary())
    s.reset()
    assert s.load()["best_score"] == 0 and s.load()["runs"] == []


def test_unwritable_location_raises_storage_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(StorageError):
        Storage(blocker / "s.json").record_run(summary())
