"""Guards for the create_ts_data mutation bug and window alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import assert_input_not_mutated, make_supervised


def test_does_not_mutate_input():
    series = pd.Series(np.arange(20, dtype=float), name="co2")
    assert_input_not_mutated(series)


def test_window_alignment():
    values = np.arange(10, dtype=float)
    X, Y = make_supervised(values, window_size=3, horizon=2)
    # First row: [0,1,2] -> targets 3 (h=1), 4 (h=2)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    np.testing.assert_array_equal(Y[0], [3, 4])
    # Last row uses the end of the series
    np.testing.assert_array_equal(X[-1], [5, 6, 7])
    np.testing.assert_array_equal(Y[-1], [8, 9])
    assert len(X) == 10 - 3 - 2 + 1


if __name__ == "__main__":
    test_does_not_mutate_input()
    test_window_alignment()
    print("window tests passed")
