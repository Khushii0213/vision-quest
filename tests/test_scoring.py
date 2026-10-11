import pytest

from src.scoring import compute_score


def test_wrong_answer_scores_zero():
    assert compute_score(False, 10, 15, 4, 150).total == 0


def test_base_with_no_time_left_no_streak():
    b = compute_score(True, 0, 15, 0)
    assert (b.base, b.speed_bonus, b.streak_bonus, b.total) == (100, 0, 0, 100)


def test_full_speed_bonus():
    assert compute_score(True, 15, 15, 0).speed_bonus == 50


def test_half_time_gives_half_bonus():
    assert compute_score(True, 7.5, 15, 0).speed_bonus == 25


def test_streak_bonus_is_capped():
    assert compute_score(True, 0, 10, 3).streak_bonus == 60
    assert compute_score(True, 0, 10, 50).streak_bonus == 100


def test_difficulty_multiplier_and_total():
    b = compute_score(True, 15, 15, 5, 150)  # (100+50+100)*1.5
    assert b.total == 375


def test_remaining_is_clamped():
    assert compute_score(True, 99, 10, 0).speed_bonus == 50
    assert compute_score(True, -5, 10, 0).speed_bonus == 0


def test_deterministic():
    assert compute_score(True, 3.3, 12, 2, 200) == compute_score(True, 3.3, 12, 2, 200)


@pytest.mark.parametrize("args", [(True, 1, 0, 0), (True, 1, 10, -1), (True, 1, 10, 0, 0)])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        compute_score(*args)
