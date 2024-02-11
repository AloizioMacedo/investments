import pickle
from pathlib import Path
from typing import List

import pandas as pd
from portfolio import TimeSeries
from raw_data_schema import RawData

DATA_FOLDER = Path("data")

PREPROCESSED_FILES = DATA_FOLDER.joinpath("02_preprocessed")
MODELS_FILES = DATA_FOLDER.joinpath("03_models")


def load_all_funds() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_FILES.joinpath("funds.csv"))


def convert_to_timeseries(df: pd.DataFrame) -> List[TimeSeries]:
    return [
        TimeSeries(df[df[RawData.CNPJ_Fundo] == cnpj].reset_index())
        for cnpj in df[RawData.CNPJ_Fundo].unique()
    ]


def main():
    funds = load_all_funds()

    timeseries = convert_to_timeseries(funds)
    with open(MODELS_FILES.joinpath("time_series.pkl"), "wb") as file:
        pickle.dump(timeseries, file)


if __name__ == "__main__":
    main()
