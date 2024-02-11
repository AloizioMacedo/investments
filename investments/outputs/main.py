import itertools
import json
import pickle
from pathlib import Path
from typing import List

import plotly.express as px
import toml
from scipy.spatial import ConvexHull

from investments.config_schema import Config
from investments.paths import CONFIG_FILE
from investments.portfolio import Portfolio, TimeSeries

DATA_FOLDER = Path("data")

MODELS_FILES = DATA_FOLDER.joinpath("03_models")
OUTPUTS_FILES = DATA_FOLDER.joinpath("04_outputs")

CONFIG = toml.load(CONFIG_FILE)
CONFIG = Config(**CONFIG)


def load_funds_timeseries() -> List[TimeSeries]:
    with open(MODELS_FILES.joinpath("time_series.pkl"), "rb") as file:
        return pickle.load(file)


def load_cdi_timeseries() -> TimeSeries:
    with open(MODELS_FILES.joinpath("cdi_ts.pkl"), "rb") as file:
        return pickle.load(file)


def main():
    time_series = load_funds_timeseries()
    cdi_time_series = load_cdi_timeseries()

    from_date = CONFIG.portfolio.from_date
    to_date = CONFIG.portfolio.to_date

    time_series.sort(key=lambda x: -x.calculate_value_at_end(from_date, to_date))

    granularity = [0.05 * i for i in range(21)]

    granularities = itertools.combinations_with_replacement(
        granularity, CONFIG.portfolio.number_of_funds
    )

    possible_splits = [gran for gran in granularities if sum(gran) == 1.0]

    best = time_series[: CONFIG.portfolio.number_of_funds]

    vols = []
    rets = []
    sharpe_ratios = []
    splits = []

    for split in possible_splits:
        split = list(split)
        p = Portfolio(best, split)

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
            for ts, split in zip(time_series, split_with_best_sharpe_ratio)
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
