from typing import List

from investments.portfolio import Portfolio, TimeSeries


def test_returned_value(five_timeseries: List[TimeSeries]):
    portfolio1 = Portfolio(five_timeseries, [0.3, 0.3, 0.4, 0.0, 0.0])
    portfolio2 = Portfolio(five_timeseries[:3], [0.3, 0.3, 0.4])

    assert portfolio1.calculate_value_at_end(
        "2020-01-01", "2021-01-01"
    ) == portfolio2.calculate_value_at_end("2020-01-01", "2021-01-01")

    portfolio1 = Portfolio(five_timeseries, [0.0, 0.3, 0.2, 0.5, 0.0])
    portfolio2 = Portfolio(five_timeseries[1:4], [0.3, 0.2, 0.5])

    assert portfolio1.calculate_value_at_end(
        "2020-01-01", "2021-01-01"
    ) == portfolio2.calculate_value_at_end("2020-01-01", "2021-01-01")
