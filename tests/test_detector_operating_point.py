from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_detector_operating_point import select_threshold, shared_probabilities  # noqa: E402


def test_shared_probabilities_reproduce_threshold_decision() -> None:
    raw = np.asarray([[0.20, 0.10], [0.80, 0.30], [0.10, 0.90]])
    probabilities = shared_probabilities(raw, 0.25)
    assert probabilities.argmax(axis=1).tolist() == [0, 1, 2]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_select_threshold_uses_only_supplied_labels_and_scores() -> None:
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    raw = np.asarray(
        [
            [0.05, 0.02],
            [0.10, 0.04],
            [0.80, 0.20],
            [0.70, 0.10],
            [0.20, 0.90],
            [0.15, 0.80],
        ]
    )
    selected, curve = select_threshold(y_true, raw, np.asarray([0.05, 0.25, 0.50]))
    assert selected == 0.25
    assert len(curve) == 3
    assert max(row["macro_f1"] for row in curve) == 1.0
