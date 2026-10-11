"""Deterministic, explainable scoring. Integer arithmetic only.

Correct answer:
    base          = 100
    speed bonus   = floor(50 * remaining / time_limit)
    streak bonus  = 20 * min(streak_before, 5)
    total         = (base + speed + streak) * difficulty_percent // 100
Wrong answer, timeout or skip: 0 points (and the streak resets).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

BASE_POINTS = 100
SPEED_MAX_BONUS = 50
STREAK_STEP_BONUS = 20
STREAK_CAP = 5


@dataclass(frozen=True)
class ScoreBreakdown:
    base: int
    speed_bonus: int
    streak_bonus: int
    multiplier_pct: int
    total: int

    def describe(self) -> str:
        if self.total == 0:
            return "0 points"
        parts = [f"base {self.base}", f"speed +{self.speed_bonus}", f"streak +{self.streak_bonus}"]
        if self.multiplier_pct != 100:
            parts.append(f"x{self.multiplier_pct / 100:g}")
        return ", ".join(parts)


ZERO = ScoreBreakdown(0, 0, 0, 100, 0)


def compute_score(correct: bool, remaining_s: float, time_limit_s: float,
                  streak_before: int, multiplier_pct: int = 100) -> ScoreBreakdown:
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if streak_before < 0:
        raise ValueError("streak_before cannot be negative")
    if multiplier_pct <= 0:
        raise ValueError("multiplier_pct must be positive")
    if not correct:
        return ZERO
    remaining = min(max(remaining_s, 0.0), time_limit_s)
    speed = math.floor(SPEED_MAX_BONUS * remaining / time_limit_s + 1e-9)
    streak = STREAK_STEP_BONUS * min(streak_before, STREAK_CAP)
    total = (BASE_POINTS + speed + streak) * multiplier_pct // 100
    return ScoreBreakdown(BASE_POINTS, speed, streak, multiplier_pct, total)
