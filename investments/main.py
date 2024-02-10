from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

DT = "dt"
VALUES = "values"


def correlation(
    ts: List[TimeSeries], from_date: dt.datetime, to_date: dt.datetime
) -> np.ndarray:
    correlation = np.corrcoef(
        [
            ts._data[(ts._data[DT] >= from_date) & (ts._data[DT] <= to_date)][VALUES]
            for ts in ts
        ]
    )

    return correlation


def get_correlation_energy(
    ts: List[TimeSeries], from_date: dt.datetime, to_date: dt.datetime
) -> float:
    cor = correlation(ts, from_date, to_date)

    diagonal_contrib = sum(x**2 for x in cor.diagonal().flatten())
    full_contrib = sum(x**2 for x in cor.flatten())
    return ((full_contrib - diagonal_contrib) / 2) ** (1 / 2)


@dataclass
class TimeSeries:
    _data: pd.DataFrame  # values, dt

    def calculate_value_at_end(
        self,
        from_date: dt.datetime,
        to_date: dt.datetime,
        initial_investiment: float = 1.0,
    ) -> float:
        values = self._data[
            (self._data[DT] >= from_date) & (self._data[DT] <= to_date)
        ][VALUES]

        product: float = values.product()  # type: ignore

        return product * initial_investiment

    def average(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self._data[
            (self._data[DT] >= from_date) & (self._data[DT] <= to_date)
        ][VALUES]

        mean = values.mean()
        return mean  # type: ignore

    def variance(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self._data[
            (self._data[DT] >= from_date) & (self._data[DT] <= to_date)
        ][VALUES]

        var: float = values.var()  # type: ignore
        return var

    def geometric_mean(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self._data[
            (self._data[DT] >= from_date) & (self._data[DT] <= to_date)
        ][VALUES]

        product: float = values.product()  # type: ignore
        geo_mean = (product) ** (1 / len(values))

        return geo_mean

    def correlation(
        self,
        other: TimeSeries,
        from_date: dt.datetime,
        to_date: dt.datetime,
    ) -> float:
        values1 = self._data[
            (self._data[DT] >= from_date) & (self._data[DT] <= to_date)
        ][VALUES]

        values2 = other._data[
            (other._data[DT] >= from_date) & (other._data[DT] <= to_date)
        ][VALUES]

        return np.corrcoef(values1, values2)[0, 1]


def main():
    df = pd.DataFrame()

    df[DT] = pd.date_range("2020-01-01", "2020-12-31")
    df[VALUES] = 1 + np.random.rand(len(df))

    ts = TimeSeries(df)

    df2 = pd.DataFrame()

    df2[DT] = pd.date_range("2020-01-01", "2020-12-31")
    df2[VALUES] = 1 + np.random.rand(len(df))

    ts2 = TimeSeries(df2)

    print(
        ts.calculate_value_at_end(
            from_date=dt.datetime(2020, 1, 1),
            to_date=dt.datetime(2020, 1, 3),
        )
    )

    print(
        ts.variance(
            from_date=dt.datetime(2020, 1, 1),
            to_date=dt.datetime(2020, 1, 3),
        )
    )

    print(
        ts.geometric_mean(
            from_date=dt.datetime(2020, 1, 1),
            to_date=dt.datetime(2020, 1, 3),
        )
    )

    print(
        ts.correlation(
            ts2,
            from_date=dt.datetime(2020, 1, 1),
            to_date=dt.datetime(2020, 1, 3),
        )
    )

    print(correlation([ts, ts2], dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 3)))

    print(
        get_correlation_energy(
            [ts, ts2], dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 3)
        )
    )


if __name__ == "__main__":
    main()
