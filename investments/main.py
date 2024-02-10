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
            ts.raw_data[(ts.raw_data[DT] >= from_date) & (ts.raw_data[DT] <= to_date)][
                VALUES
            ]
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
    raw_data: pd.DataFrame  # values, dt

    def calculate_value_at_end(
        self,
        from_date: dt.datetime,
        to_date: dt.datetime,
        initial_investiment: float = 1.0,
    ) -> float:
        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        product: float = values.product()  # type: ignore

        return product * initial_investiment

    def average(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        mean = values.mean()
        return mean  # type: ignore

    def variance(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        var: float = values.var()  # type: ignore
        return var

    def geometric_mean(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
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
        values1 = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        values2 = other.raw_data[
            (other.raw_data[DT] >= from_date) & (other.raw_data[DT] <= to_date)
        ][VALUES]

        return np.corrcoef(values1, values2)[0, 1]


class Portfolio:
    def __init__(self, time_series: List[TimeSeries], split: List[float]):
        self.time_series = time_series

        if len(split) != len(time_series):
            raise ValueError("Split must have same length as time series")

        if sum(split) != 1:
            raise ValueError("Split must sum to 1")

        self.split = split

    def correlation_energy(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        return get_correlation_energy(self.time_series, from_date, to_date)

    def correlation(self, from_date: dt.datetime, to_date: dt.datetime) -> np.ndarray:
        return correlation(self.time_series, from_date, to_date)

    def calculate_value_at_end(
        self,
        from_date: dt.datetime,
        to_date: dt.datetime,
        initial_investiment: float = 1.0,
    ) -> float:
        return sum(
            [
                ts.calculate_value_at_end(
                    from_date, to_date, initial_investiment * split
                )
                for split, ts in zip(self.split, self.time_series)
            ]
        )


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
