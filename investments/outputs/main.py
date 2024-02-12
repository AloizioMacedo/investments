import itertools
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import plotly.express as px
from scipy.spatial import ConvexHull
from tqdm import tqdm

from investments.config import CONFIG
from investments.portfolio import Portfolio, TimeSeries

DATA_FOLDER = Path("data")

MODELS_FILES = DATA_FOLDER.joinpath("03_models")
OUTPUTS_FILES = DATA_FOLDER.joinpath("04_outputs")


def load_funds_timeseries() -> List[TimeSeries]:
    with open(MODELS_FILES.joinpath("time_series.pkl"), "rb") as file:
        return pickle.load(file)


def load_cdi_timeseries() -> TimeSeries:
    with open(MODELS_FILES.joinpath("cdi_ts.pkl"), "rb") as file:
        return pickle.load(file)


def get_possible_splits() -> List[Tuple[float, ...]]:
    granularity = [
        round(CONFIG.portfolio.split_granularity * i, 5)
        for i in range(round(1.0 / CONFIG.portfolio.split_granularity) + 1)
    ]

    granularities = itertools.product(
        granularity, repeat=CONFIG.portfolio.number_of_funds
    )

    possible_splits = [gran for gran in granularities if sum(gran) == 1.0]
    return possible_splits


def get_best_funds(from_date: str, to_date: str):
    time_series = load_funds_timeseries()

    time_series.sort(key=lambda x: -x.calculate_value_at_end(from_date, to_date))
    best = time_series[: CONFIG.portfolio.number_of_funds]
    return best


def main():
    cdi_time_series = load_cdi_timeseries()

    from_date = CONFIG.portfolio.from_date
    to_date = CONFIG.portfolio.to_date

    best_funds = get_best_funds(from_date, to_date)

    possible_splits = get_possible_splits()

    vols = []
    avgs = []
    rets_at_end = []
    sharpe_ratios = []
    splits = []

    for split in tqdm(
        possible_splits, desc="Calculating portfolios based on granularity"
    ):
        split = list(split)
        p = Portfolio(best_funds, cdi_time_series, split, from_date, to_date)

        vols.append(p.std_dev())
        avgs.append(p.average() - 1)
        rets_at_end.append(p.calculate_value_at_end(from_date, to_date))
        sharpe_ratios.append(p.sharpe_ratio())
        splits.append(split)

    efficient_frontier = px.scatter(x=vols, y=avgs, hover_name=splits)

    efficient_frontier.write_html(OUTPUTS_FILES.joinpath("efficient_frontier.html"))
    efficient_frontier.write_image(OUTPUTS_FILES.joinpath("efficient_frontier.png"))

    risk_return = px.scatter(x=vols, y=rets_at_end, hover_name=splits)

    risk_return.write_html(OUTPUTS_FILES.joinpath("risk_return.html"))
    risk_return.write_image(OUTPUTS_FILES.joinpath("risk_return.png"))

    points = list(zip(vols, avgs))
    ch = ConvexHull(points)

    x_hull = [points[i][0] for i in ch.vertices]
    y_hull = [points[i][1] for i in ch.vertices]
    ch_sharpe_ratios = [sharpe_ratios[i] for i in ch.vertices]
    ch_splits = [splits[i] for i in ch.vertices]

    convex_hull = px.scatter(x=x_hull, y=y_hull, hover_name=ch_splits)
    convex_hull.write_html(OUTPUTS_FILES.joinpath("convex_hull.html"))
    convex_hull.write_image(OUTPUTS_FILES.joinpath("convex_hull.png"))

    vol, ret, ret_at_end, best_ratio, split_with_best_sharpe_ratio = max(
        (
            (vol, ret, ret_at_end, ratio, split)
            for vol, ret, ret_at_end, ratio, split in zip(
                x_hull, y_hull, rets_at_end, ch_sharpe_ratios, ch_splits
            )
        ),
        key=lambda x: x[3],
    )

    best_allocation = {
        "allocations": {
            ts.cnpj: split
            for ts, split in zip(best_funds, split_with_best_sharpe_ratio)
        },
        "sharpe_ratio": best_ratio,
        "expected_returns_at_end": ret_at_end,
        "average": ret,
        "volatility": vol,
    }

    with open(
        DATA_FOLDER.joinpath("05_reporting").joinpath("best_allocation.json"), "w"
    ) as file:
        json.dump(best_allocation, file)


if __name__ == "__main__":
    main()
