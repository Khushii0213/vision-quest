import pytest

from src.game import DIFFICULTIES, Game, GameError, Outcome, Phase, grid_columns


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(diff="normal", classes=10, per=12, seed=1):
    index = {c: list(range(c * per, (c + 1) * per)) for c in range(classes)}
    clock = Clock()
    return Game(DIFFICULTIES[diff], index, seed=seed, clock=clock), clock


def test_round_structure_and_unique_tiles():
    for diff, cfg in DIFFICULTIES.items():
        g, _ = make(diff)
        r = g.round
        assert len(r.tiles) == cfg.tile_count(1)
        assert len({t.dataset_index for t in r.tiles}) == len(r.tiles)
        assert r.tiles[r.correct_position].label == r.target_label
        assert sum(t.label == r.target_label for t in r.tiles) == 1
        assert r.reference_index not in {t.dataset_index for t in r.tiles}


def test_ground_truth_comes_from_labels():
    g, _ = make()
    t = g.round.tiles[g.round.correct_position]
    assert t.dataset_index // 12 == g.round.target_label


def test_correct_answer_scores_once_and_blocks_second_submit():
    g, c = make()
    c.t += 3
    res = g.submit(g.round.correct_position)
    assert res.outcome is Outcome.CORRECT and res.score.total > 0
    score = g.score
    assert g.phase is Phase.REVEAL
    with pytest.raises(GameError):
        g.submit(g.round.correct_position)
    with pytest.raises(GameError):
        g.skip()
    assert g.score == score and len(g.results) == 1


def test_wrong_answer_zero_points_resets_streak_costs_life():
    g, _ = make()
    g.submit(g.round.correct_position)
    g.next_round()
    assert g.streak == 1
    wrong = (g.round.correct_position + 1) % len(g.round.tiles)
    lives = g.lives
    res = g.submit(wrong)
    assert res.outcome is Outcome.WRONG and res.score.total == 0
    assert g.streak == 0 and g.best_streak == 1 and g.lives == lives - 1


def test_streak_bonus_applies_next_round():
    g, _ = make()
    first = g.submit(g.round.correct_position)
    g.next_round()
    second = g.submit(g.round.correct_position)
    assert first.score.streak_bonus == 0 and second.score.streak_bonus == 20


def test_timer_counts_down_and_expires():
    g, c = make()
    limit = g.round.time_limit
    assert g.remaining() == pytest.approx(limit)
    c.t += 5
    assert g.remaining() == pytest.approx(limit - 5)
    assert g.expire() is None and g.phase is Phase.PLAYING
    c.t += limit
    res = g.expire()
    assert res.outcome is Outcome.TIMEOUT and g.remaining() == 0 and g.phase is Phase.REVEAL
    with pytest.raises(GameError):
        g.expire()


def test_late_click_does_not_score():
    g, c = make()
    c.t += g.round.time_limit + 0.1
    res = g.submit(g.round.correct_position)
    assert res.outcome is Outcome.TIMEOUT and g.score == 0


def test_skip_limits_and_effects():
    g, _ = make("hard")  # 1 skip
    g.streak = 3
    g.skip()
    assert g.streak == 0 and g.skips_left == 0 and g.lives == DIFFICULTIES["hard"].lives
    g.next_round()
    with pytest.raises(GameError):
        g.skip()


def test_invalid_position():
    g, _ = make()
    for pos in (-1, 99):
        with pytest.raises(GameError):
            g.submit(pos)
    assert g.phase is Phase.PLAYING


def test_game_over_out_of_lives():
    g, _ = make("hard")  # 2 lives
    for _ in range(2):
        g.submit((g.round.correct_position + 1) % len(g.round.tiles))
        g.next_round()
    assert g.phase is Phase.OVER and g.over_reason == "out_of_lives"
    with pytest.raises(GameError):
        g.submit(0)


def test_run_completes_after_configured_rounds():
    g, _ = make("easy")
    for _ in range(DIFFICULTIES["easy"].rounds):
        assert g.phase is Phase.PLAYING
        g.submit(g.round.correct_position)
        g.next_round()
    assert g.phase is Phase.OVER and g.over_reason == "completed"
    s = g.summary()
    assert s["correct"] == 10 and s["rounds_played"] == 10 and s["best_streak"] == 10 and s["score"] == g.score


def test_next_round_requires_reveal():
    g, _ = make()
    with pytest.raises(GameError):
        g.next_round()


def test_progressive_difficulty_is_monotonic():
    for cfg in DIFFICULTIES.values():
        times = [cfg.time_limit(n) for n in range(1, cfg.rounds + 1)]
        strengths = [cfg.strength(n) for n in range(1, cfg.rounds + 1)]
        tiles = [cfg.tile_count(n) for n in range(1, cfg.rounds + 1)]
        assert times == sorted(times, reverse=True) and times[-1] >= cfg.time_floor
        assert strengths == sorted(strengths) and strengths[-1] <= cfg.strength_max
        assert tiles == sorted(tiles)


def test_difficulty_ordering():
    e, n, h = (DIFFICULTIES[k] for k in ("easy", "normal", "hard"))
    assert e.time_start > n.time_start > h.time_start
    assert e.tile_count(1) < n.tile_count(1) < h.tile_count(1)
    assert e.multiplier_pct < n.multiplier_pct < h.multiplier_pct


def test_more_tiles_than_other_classes_still_unique():
    g, _ = make("hard", classes=3, per=30)  # 7 distractors drawn from only 2 other classes
    assert len(g.round.tiles) == 8 and len({t.dataset_index for t in g.round.tiles}) == 8


def test_largest_grid_has_unique_tiles():
    g, _ = make("hard", classes=4, per=30, seed=5)
    while g.round.number < DIFFICULTIES["hard"].grow_at_round:
        g.submit(g.round.correct_position)
        g.next_round()
    assert len(g.round.tiles) == 12 and len({t.dataset_index for t in g.round.tiles}) == 12


def test_seeded_games_are_reproducible():
    a, _ = make(seed=7)
    b, _ = make(seed=7)
    assert a.round.tiles == b.round.tiles and a.round.correct_position == b.round.correct_position


def test_unusable_dataset_rejected():
    with pytest.raises(GameError):
        Game(DIFFICULTIES["easy"], {0: [1, 2, 3]})
    with pytest.raises(GameError):
        Game(DIFFICULTIES["easy"], {})


def test_abandon_marks_recorded():
    g, _ = make()
    g.abandon()
    assert g.phase is Phase.OVER and g.recorded and g.over_reason == "abandoned"


@pytest.mark.parametrize("n,cols", [(4, 2), (6, 3), (8, 4), (9, 3), (12, 4)])
def test_grid_columns(n, cols):
    assert grid_columns(n) == cols
