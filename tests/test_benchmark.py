from __future__ import annotations

from blc_inaspeech.schema import Segment
from scripts.benchmark import canonical, measure


def test_benchmark_retains_every_sample_for_label_validation() -> None:
    counter = 0

    def operation() -> int:
        nonlocal counter
        counter += 1
        return counter

    seconds, results = measure(operation, 3)
    assert len(seconds) == 3 and all(value >= 0 for value in seconds)
    assert results == [1, 2, 3]


def test_gender_baseline_normalization_preserves_speech_boundaries() -> None:
    result = canonical([("female", 0, 0.2), ("male", 0.2, 0.5), ("music", 0.5, 1)], 1.0)
    assert result == (Segment("speech", 0, 0.5), Segment("music", 0.5, 1))
