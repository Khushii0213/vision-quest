"""Arcade game engine: pure Python, no Streamlit, fully unit-testable.

State machine:  PLAYING --submit/skip/expire--> REVEAL --next_round--> PLAYING | OVER
Only PLAYING accepts an answer, so a round can never be scored twice.
Ground truth comes from the dataset labels (``index_by_class``), never from the AI model.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

from .augmentation import AugmentParams, random_params
from .scoring import ScoreBreakdown, ZERO, compute_score


class Phase(str, Enum):
    PLAYING = "playing"
    REVEAL = "reveal"
    OVER = "over"


class Outcome(str, Enum):
    CORRECT = "correct"
    WRONG = "wrong"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class GameError(RuntimeError):
    """Raised for illegal actions (e.g. answering twice) or unusable data."""


@dataclass(frozen=True)
class DifficultyConfig:
    key: str
    label: str
    rounds: int
    lives: int
    skips: int
    tiles_start: int
    tiles_max: int
    grow_at_round: int       # round number from which tiles_max is used
    time_start: float
    time_floor: float
    time_decay: float        # seconds removed per round
    strength_start: float
    strength_max: float
    strength_step: float
    multiplier_pct: int

    def tile_count(self, n: int) -> int:
        return self.tiles_max if n >= self.grow_at_round else self.tiles_start

    def time_limit(self, n: int) -> float:
        return max(self.time_floor, self.time_start - self.time_decay * (n - 1))

    def strength(self, n: int) -> float:
        return min(self.strength_max, self.strength_start + self.strength_step * (n - 1))


DIFFICULTIES: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig("easy", "Easy", rounds=10, lives=5, skips=3, tiles_start=4, tiles_max=6,
                             grow_at_round=6, time_start=20, time_floor=12, time_decay=0.8,
                             strength_start=0.15, strength_max=0.5, strength_step=0.04, multiplier_pct=100),
    "normal": DifficultyConfig("normal", "Normal", rounds=12, lives=3, skips=2, tiles_start=6, tiles_max=9,
                               grow_at_round=6, time_start=15, time_floor=8, time_decay=0.6,
                               strength_start=0.30, strength_max=0.75, strength_step=0.04, multiplier_pct=150),
    "hard": DifficultyConfig("hard", "Hard", rounds=15, lives=2, skips=1, tiles_start=8, tiles_max=12,
                             grow_at_round=6, time_start=12, time_floor=5, time_decay=0.5,
                             strength_start=0.50, strength_max=1.0, strength_step=0.05, multiplier_pct=200),
}


def grid_columns(n: int) -> int:
    """Pick a tidy column count: prefers 4, then 3, then 2 columns with at least two rows."""
    for cols in (4, 3, 2):
        if n % cols == 0 and n // cols >= 2:
            return cols
    return min(n, 3)


@dataclass(frozen=True)
class Tile:
    dataset_index: int
    label: int
    params: AugmentParams
    seed: int


@dataclass(frozen=True)
class Round:
    number: int
    target_label: int
    reference_index: int
    tiles: tuple[Tile, ...]
    correct_position: int
    time_limit: float
    started_at: float
    deadline: float


@dataclass(frozen=True)
class RoundResult:
    number: int
    outcome: Outcome
    chosen_position: int | None
    correct_position: int
    response_time: float
    score: ScoreBreakdown


class Game:
    MIN_IMAGES_PER_CLASS = 5

    def __init__(self, difficulty: DifficultyConfig, index_by_class: Mapping[int, Sequence[int]],
                 *, seed: int | None = None, clock: Callable[[], float] = time.monotonic):
        usable = {c: list(v) for c, v in index_by_class.items() if len(v) >= self.MIN_IMAGES_PER_CLASS}
        if len(usable) < 2:
            raise GameError("The dataset needs at least two classes with 5+ images each.")
        self.difficulty = difficulty
        self._classes = usable
        self._rng = random.Random(seed)
        self._clock = clock
        self.phase = Phase.PLAYING
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.lives = difficulty.lives
        self.skips_left = difficulty.skips
        self.results: list[RoundResult] = []
        self.over_reason: str | None = None
        self.recorded = False
        self.round: Round
        self._start_round(1)

    # ---- helpers -------------------------------------------------------
    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _require(self, phase: Phase) -> None:
        if self.phase is not phase:
            raise GameError(f"Action not allowed in phase '{self.phase.value}' (needs '{phase.value}').")

    def _start_round(self, number: int) -> None:
        d, rng = self.difficulty, self._rng
        n_tiles = d.tile_count(number)
        strength = d.strength(number)
        target = rng.choice(sorted(self._classes))
        correct_idx, reference_idx = rng.sample(self._classes[target], 2)

        others = sorted(c for c in self._classes if c != target)
        rng.shuffle(others)
        picks = [others[i % len(others)] for i in range(n_tiles - 1)]
        per_class: dict[int, int] = {}
        for c in picks:
            per_class[c] = per_class.get(c, 0) + 1
        chosen: list[tuple[int, int]] = []
        for c, count in per_class.items():
            pool = [i for i in self._classes[c] if i not in (correct_idx, reference_idx)]
            if len(pool) < count:
                raise GameError("Not enough distinct images to build a round.")
            chosen += [(i, c) for i in rng.sample(pool, count)]

        raw = [(correct_idx, target)] + chosen
        tiles = [Tile(idx, lab, random_params(rng, strength), rng.getrandbits(32)) for idx, lab in raw]
        order = list(range(len(tiles)))
        rng.shuffle(order)
        tiles = [tiles[i] for i in order]
        correct_pos = order.index(0)
        assert len({t.dataset_index for t in tiles}) == len(tiles), "duplicate tiles"

        limit = d.time_limit(number)
        start = self._clock()
        self.round = Round(number, target, reference_idx, tuple(tiles), correct_pos, limit, start, start + limit)
        self.phase = Phase.PLAYING

    def _finish(self, outcome: Outcome, chosen: int | None, now: float) -> RoundResult:
        r = self.round
        remaining = max(0.0, r.deadline - now)
        elapsed = min(max(now - r.started_at, 0.0), r.time_limit)
        if outcome is Outcome.CORRECT:
            breakdown = compute_score(True, remaining, r.time_limit, self.streak, self.difficulty.multiplier_pct)
            self.score += breakdown.total
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
        else:
            breakdown = ZERO
            self.streak = 0
            if outcome in (Outcome.WRONG, Outcome.TIMEOUT):
                self.lives -= 1
            else:
                self.skips_left -= 1
        result = RoundResult(r.number, outcome, chosen, r.correct_position, elapsed, breakdown)
        self.results.append(result)
        self.phase = Phase.REVEAL
        return result

    # ---- public actions --------------------------------------------------
    def remaining(self, now: float | None = None) -> float:
        if self.phase is not Phase.PLAYING:
            return 0.0
        return max(0.0, self.round.deadline - self._now(now))

    def submit(self, position: int, now: float | None = None) -> RoundResult:
        self._require(Phase.PLAYING)
        if not 0 <= position < len(self.round.tiles):
            raise GameError(f"Tile position {position} is out of range.")
        t = self._now(now)
        if t >= self.round.deadline:  # a click that arrives after the deadline does not count
            return self._finish(Outcome.TIMEOUT, None, t)
        outcome = Outcome.CORRECT if position == self.round.correct_position else Outcome.WRONG
        return self._finish(outcome, position, t)

    def expire(self, now: float | None = None) -> RoundResult | None:
        """End the round if its time is up. Returns None if there is still time left."""
        self._require(Phase.PLAYING)
        t = self._now(now)
        if t < self.round.deadline:
            return None
        return self._finish(Outcome.TIMEOUT, None, t)

    def skip(self, now: float | None = None) -> RoundResult:
        self._require(Phase.PLAYING)
        if self.skips_left <= 0:
            raise GameError("No skips left.")
        return self._finish(Outcome.SKIPPED, None, self._now(now))

    def next_round(self) -> None:
        self._require(Phase.REVEAL)
        if self.lives <= 0:
            self.phase, self.over_reason = Phase.OVER, "out_of_lives"
        elif self.round.number >= self.difficulty.rounds:
            self.phase, self.over_reason = Phase.OVER, "completed"
        else:
            self._start_round(self.round.number + 1)

    def abandon(self) -> None:
        if self.phase is not Phase.OVER:
            self.phase, self.over_reason = Phase.OVER, "abandoned"
            self.recorded = True  # abandoned runs are not saved to the history

    @property
    def last_result(self) -> RoundResult | None:
        return self.results[-1] if self.results else None

    def summary(self) -> dict:
        answered = [r for r in self.results if r.outcome in (Outcome.CORRECT, Outcome.WRONG)]
        count = lambda o: sum(1 for r in self.results if r.outcome is o)  # noqa: E731
        return {
            "difficulty": self.difficulty.key,
            "score": self.score,
            "correct": count(Outcome.CORRECT),
            "wrong": count(Outcome.WRONG),
            "timeouts": count(Outcome.TIMEOUT),
            "skipped": count(Outcome.SKIPPED),
            "rounds_played": len(self.results),
            "answered": len(answered),
            "response_time_total": round(sum(r.response_time for r in answered), 3),
            "avg_response_s": round(sum(r.response_time for r in answered) / len(answered), 2) if answered else None,
            "best_streak": self.best_streak,
            "end_reason": self.over_reason,
        }
