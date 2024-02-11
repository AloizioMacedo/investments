from typing import List

import numpy as np
import pandas as pd
import pytest

from investments.portfolio import TimeSeries

np.random.seed(42)


@pytest.fixture
def timeseries() -> TimeSeries:
    dt_range = pd.date_range("2020-01-01", "2021-01-01")

    df = pd.DataFrame()

    df["dt"] = dt_range
    df["values"] = np.random.normal(1, 0.01, len(dt_range))

    return TimeSeries(df, "tf1")


@pytest.fixture
def five_timeseries() -> List[TimeSeries]:
    dt_range = pd.date_range("2020-01-01", "2021-01-01")

    ts = []
    for i in range(5):
        df = pd.DataFrame()

        df["dt"] = dt_range
        df["values"] = np.random.normal(1, 0.01, len(dt_range))

        ts.append(TimeSeries(df, f"tf{i}"))

    return ts
