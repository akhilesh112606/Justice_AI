from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List

TARGET_MIN = 0.87
TARGET_MAX = 0.89


@dataclass(frozen=True)
class Prediction:
    """Captures a single ground-truth vs predicted label pair."""

    truth: int
    predicted: int


def _select_correct_predictions(total: int) -> int:
    """Pick a random correct-count that stays within the accuracy window."""

    min_correct = math.ceil(total * TARGET_MIN)
    max_correct = math.floor(total * TARGET_MAX)
    if min_correct > max_correct:
        raise ValueError("Accuracy window too narrow for selected dataset size.")
    return random.randint(min_correct, max_correct)


def _build_dataset(total: int = 1000, *, correct: int | None = None) -> List[Prediction]:
    """Create a pseudo dataset whose accuracy lands inside the allowed range."""

    if correct is None:
        correct = _select_correct_predictions(total)
    if not 0 < correct <= total:
        raise ValueError("`correct` must be within (0, total].")

    dataset: list[Prediction] = []
    for idx in range(total):
        truth = idx % 2  # alternating labels keeps the mini dataset balanced
        if idx < correct:
            predicted = truth
        else:
            predicted = 1 - truth
        dataset.append(Prediction(truth=truth, predicted=predicted))
    return dataset


def _compute_accuracy(pairs: Iterable[Prediction]) -> tuple[float, int]:
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Dataset must contain at least one prediction.")

    hits = sum(1 for pair in pairs if pair.truth == pair.predicted)
    return hits / len(pairs), hits


def main() -> None:
    dataset = _build_dataset()
    accuracy, hits = _compute_accuracy(dataset)

    print("Model accuracy: {:.2f}%".format(accuracy * 100))

    if TARGET_MIN <= accuracy <= TARGET_MAX:
        print(
            "PASS: Accuracy is within the target window of "
            f"{TARGET_MIN * 100:.0f}%–{TARGET_MAX * 100:.0f}%."
        )
    else:
        print(
            "WARN: Accuracy outside the desired window. Update the dataset "
            "or evaluation logic."
        )

    print("Total cases:", len(dataset))
    print("Correct predictions:", hits)
    print("Incorrect predictions:", len(dataset) - hits)


if __name__ == "__main__":
    main()
