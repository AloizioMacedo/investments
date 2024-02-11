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
    rets = []
    sharpe_ratios = []
    splits = []

    for split in tqdm(
        possible_splits, desc="Calculating portfolios based on granularity"
    ):
        split = list(split)
        p = Portfolio(best_funds, split)

        vols.append(p.std_dev(from_date, to_date))
        rets.append(p.calculate_value_at_end(from_date, to_date))
        sharpe_ratios.append(p.sharpe_ratio(cdi_time_series, from_date, to_date))
        splits.append(split)

    fig = px.scatter(x=vols, y=rets, hover_name=splits)

    fig.write_html(OUTPUTS_FILES.joinpath("risk_return.html"))
    fig.write_image(OUTPUTS_FILES.joinpath("risk_return.png"))

    points = list(zip(vols, rets))
    ch = ConvexHull(points)

    x_hull = [points[i][0] for i in ch.vertices]
    y_hull = [points[i][1] for i in ch.vertices]
    ch_sharpe_ratios = [sharpe_ratios[i] for i in ch.vertices]
    ch_splits = [splits[i] for i in ch.vertices]

    fig2 = px.scatter(x=x_hull, y=y_hull, hover_name=ch_splits)
    fig2.write_html(OUTPUTS_FILES.joinpath("convex_hull.html"))
    fig2.write_image(OUTPUTS_FILES.joinpath("convex_hull.png"))

    vol, ret, best_ratio, split_with_best_sharpe_ratio = max(
        (
            (vol, ret, ratio, split)
            for vol, ret, ratio, split in zip(
                x_hull, y_hull, ch_sharpe_ratios, ch_splits
            )
        ),
        key=lambda x: x[2],
    )

    best_allocation = {
        "allocations": {
            ts.cnpj: split
            for ts, split in zip(best_funds, split_with_best_sharpe_ratio)
        },
        "sharpe_ratio": best_ratio,
        "expected_returns": ret,
        "expected_volatility": vol,
    }

    with open(
        DATA_FOLDER.joinpath("05_reporting").joinpath("best_allocation.json"), "w"
    ) as file:
        json.dump(best_allocation, file)


if __name__ == "__main__":
    main()
