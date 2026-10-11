"""Compare two top-k prediction lists (original vs transformed image). Pure Python."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonRow:
    class_index: int
    label: str
    original_rank: int | None
    original_conf: float | None
    augmented_rank: int | None
    augmented_conf: float | None
    status: str  # "same" | "moved" | "entered" | "dropped"

    @property
    def delta(self) -> float | None:
        if self.original_conf is None or self.augmented_conf is None:
            return None
        return self.augmented_conf - self.original_conf


@dataclass(frozen=True)
class Comparison:
    rows: tuple[ComparisonRow, ...]
    top1_changed: bool
    original_top1: object
    augmented_top1: object

    def describe(self) -> str:
        o, a = self.original_top1, self.augmented_top1
        if self.top1_changed:
            head = (f"The top-1 prediction changed from '{o.label}' ({o.confidence:.1%}) "
                    f"to '{a.label}' ({a.confidence:.1%}).")
        else:
            head = (f"The top-1 prediction stayed '{o.label}'; its confidence went from "
                    f"{o.confidence:.1%} to {a.confidence:.1%}.")
        n_changed = sum(1 for r in self.rows if r.status != "same")
        return head + f" {n_changed} of {len(self.rows)} top-5 entries changed rank or membership."


def compare_predictions(original, augmented) -> Comparison:
    if not original or not augmented:
        raise ValueError("Both prediction lists must be non-empty.")
    orig = {p.class_index: (i + 1, p) for i, p in enumerate(original)}
    aug = {p.class_index: (i + 1, p) for i, p in enumerate(augmented)}
    rows = []
    for idx in list(orig) + [i for i in aug if i not in orig]:
        o, a = orig.get(idx), aug.get(idx)
        label = (o or a)[1].label
        if o and a:
            status = "same" if o[0] == a[0] else "moved"
        else:
            status = "dropped" if o else "entered"
        rows.append(ComparisonRow(idx, label, o[0] if o else None, o[1].confidence if o else None,
                                  a[0] if a else None, a[1].confidence if a else None, status))
    return Comparison(tuple(rows), original[0].class_index != augmented[0].class_index, original[0], augmented[0])
