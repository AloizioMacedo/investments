from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DT = "dt"
VALUES = "values"


@dataclass
class TimeSeries:
    def __init__(
        self,
        raw_data: pd.DataFrame,
        cnpj: str = "",
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ):
        self.raw_data = raw_data
        self.cnpj = cnpj

        min_date = self.raw_data[DT].min()
        max_date = self.raw_data[DT].max()

        self._min_date = min_date if min_date is not None else min_date
        self._max_date = max_date if max_date is not None else max_date

        self._calculation_cache: Dict[Tuple[str, str], float] = {}

    def filter_ts(self, from_date: str, to_date: str):
        if self._min_date == from_date and self._max_date == to_date:
            return

        self.raw_data = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ].reset_index()

        self._min_date = from_date
        self._max_date = to_date

    def calculate_value_at_end(
        self,
        from_date: str,
        to_date: str,
        initial_investiment: float = 1.0,
    ) -> float:
        if (result := self._calculation_cache.get((from_date, to_date))) is not None:
            return result * initial_investiment

        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        product: float = values.product()  # type: ignore

        self._calculation_cache[(from_date, to_date)] = product

        return product * initial_investiment

    def average(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
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
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> float:
        from_date = from_date if from_date is not None else self._min_date
        to_date = to_date if to_date is not None else self._max_date

        values = self.raw_data[
            (self.raw_data[DT] >= from_date) & (self.raw_data[DT] <= to_date)
        ][VALUES]

        var: float = values.var()  # type: ignore
        return var

    def geometric_mean(self, from_date: Optional[str], to_date: Optional[str]) -> float:
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
        from_date: Optional[str],
        to_date: Optional[str],
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
    def __init__(
        self,
        time_series: List[TimeSeries],
        split: List[float],
    ):
        self.time_series = time_series

        if len(split) != len(time_series):
            raise ValueError("Split must have same length as time series")

        if sum(split) != 1:
            raise ValueError("Split must sum to 1")

        self.split = split

    def std_dev(self) -> float:
        portfolio_ts: pd.Series = sum(
            split * ts.raw_data[VALUES]
            for split, ts in zip(self.split, self.time_series)
        )  # type: ignore

        return portfolio_ts.std()

    def average(self) -> float:
        portfolio_ts: pd.Series = sum(
            split * ts.raw_data[VALUES]
            for split, ts in zip(self.split, self.time_series)
        )  # type: ignore

        return portfolio_ts.mean()

    def sharpe_ratio(self, risk_free: TimeSeries) -> float:
        portfolio_ts: pd.Series = sum(
            split * ts.raw_data[VALUES]
            for split, ts in zip(self.split, self.time_series)
        )  # type: ignore

        risk_free_ts: pd.Series = risk_free.raw_data[VALUES]

        excess = portfolio_ts - risk_free_ts

        copied_df = risk_free.raw_data.reset_index(drop=True)
        copied_df[VALUES] = excess

        excess_ts = TimeSeries(copied_df)
        std = (excess_ts.variance()) ** (1 / 2)
        avg = excess_ts.average()

        return avg / std

    def calculate_value_at_end(
        self,
        from_date: str,
        to_date: str,
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
