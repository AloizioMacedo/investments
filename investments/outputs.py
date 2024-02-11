import itertools
import pickle
from pathlib import Path
from typing import List

import plotly.express as px
import toml
from config_schema import Config
from paths import CONFIG_FILE
from portfolio import Portfolio, TimeSeries

DATA_FOLDER = Path("data")

MODELS_FILES = DATA_FOLDER.joinpath("03_models")
OUTPUTS_FILES = DATA_FOLDER.joinpath("04_outputs")

CONFIG = toml.load(CONFIG_FILE)
CONFIG = Config(**CONFIG)


def load_timeseries() -> List[TimeSeries]:
    with open(MODELS_FILES.joinpath("time_series.pkl"), "rb") as file:
        return pickle.load(file)


def main():
    time_series = load_timeseries()

    from_date = CONFIG.portfolio.from_date
    to_date = CONFIG.portfolio.to_date

    time_series.sort(key=lambda x: -x.calculate_value_at_end(from_date, to_date))

    granularity = [0.05 * i for i in range(21)]

    possible_granularities = itertools.combinations_with_replacement(
        granularity, CONFIG.portfolio.number_of_funds
    )

    possible_granularities = [
        gran for gran in possible_granularities if sum(gran) == 1.0
    ]

    best = time_series[: CONFIG.portfolio.number_of_funds]

    vols = []
    rets = []
    grans = []
    for gran in possible_granularities:
        gran = list(gran)
        p = Portfolio(best, gran)

        vols.append(p.std_dev(from_date, to_date))
        rets.append(p.calculate_value_at_end(from_date, to_date))
        grans.append(gran)

    fig = px.scatter(x=vols, y=rets, hover_name=grans)

    fig.write_html(OUTPUTS_FILES.joinpath("scatter.html"))
    fig.write_image(OUTPUTS_FILES.joinpath("scatter.png"))


if __name__ == "__main__":
    main()
