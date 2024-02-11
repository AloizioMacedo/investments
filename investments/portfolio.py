from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass
from typing import List, Optional

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

        self._min_date = values[DT].min()
        self._max_date = values[DT].max()

        return product * initial_investiment

    def average(
        self,
        from_date: Optional[dt.datetime] = None,
        to_date: Optional[dt.datetime] = None,
    ) -> float:
        from_date = from_date if from_date is not None else self._min_date
        to_date = to_date if to_date is not None else self._max_date

        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        mean = values.mean()
        return mean  # type: ignore

    def variance(
        self,
        from_date: Optional[dt.datetime] = None,
        to_date: Optional[dt.datetime] = None,
    ) -> float:
        from_date = from_date if from_date is not None else self._min_date
        to_date = to_date if to_date is not None else self._max_date

        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        var: float = values.var()  # type: ignore
        return var

    def geometric_mean(
        self, from_date: Optional[dt.datetime], to_date: Optional[dt.datetime]
    ) -> float:
        from_date = from_date if from_date is not None else self._min_date
        to_date = to_date if to_date is not None else self._max_date

        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        product: float = values.product()  # type: ignore
        geo_mean = (product) ** (1 / len(values))

        return geo_mean

    def correlation(
        self,
        other: TimeSeries,
        from_date: Optional[dt.datetime],
        to_date: Optional[dt.datetime],
    ) -> float:
        from_date = from_date if from_date is not None else self._min_date
        to_date = to_date if to_date is not None else self._max_date

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

    def cost(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        return (
            self.std_dev(from_date, to_date)
            + 10 * self.correlation_energy(from_date, to_date)
            - 1 * self.calculate_value_at_end(from_date, to_date)
        )

    def std_dev(self, from_date: dt.datetime, to_date: dt.datetime) -> float:
        return sum(ts.variance(from_date, to_date) for ts in self.time_series)

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
    df[VALUES] = 1 + np.random.normal(0, 0.1, len(df))

    ts = TimeSeries(df)

    dfs = []

    for _ in range(10):
        df = pd.DataFrame()
        df[DT] = pd.date_range("2020-01-01", "2020-12-31")
        df[VALUES] = 1 + np.random.normal(0, 0.1, len(df))

        dfs.append(df)

    combinations = itertools.combinations(dfs, 2)

    all_ts = [TimeSeries(x) for x in dfs]

    portfolios = [
        Portfolio([TimeSeries(x) for x in comb], [0.5, 0.5]) for comb in combinations
    ]

    best_portfolio = min(
        portfolios,
        key=lambda x: x.cost(
            from_date=dt.datetime(2020, 1, 1), to_date=dt.datetime(2020, 12, 31)
        ),
    )

    ts1, ts2 = best_portfolio.time_series

    ts1.raw_data.to_csv("ts1.csv")
    ts2.raw_data.to_csv("ts2.csv")

    print(
        f"Best: {ts1.calculate_value_at_end(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))}"
    )
    print(
        f"Best: {ts2.calculate_value_at_end(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))}"
    )

    for ts in all_ts:
        print(
            f"Listing others: {ts.calculate_value_at_end(dt.datetime(2020, 1, 1), dt.datetime(2020, 12, 31))}"
        )


if __name__ == "__main__":
    main()
