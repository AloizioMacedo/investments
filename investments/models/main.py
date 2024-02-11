import pickle
from pathlib import Path
from typing import List

import pandas as pd

from investments.portfolio import TimeSeries
from investments.preprocess.raw_data_schema import RawData

DATA_FOLDER = Path("data")

PREPROCESSED_FILES = DATA_FOLDER.joinpath("02_preprocessed")
MODELS_FILES = DATA_FOLDER.joinpath("03_models")


def load_all_funds() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_FILES.joinpath("funds.csv"))


def convert_to_timeseries(df: pd.DataFrame) -> List[TimeSeries]:
    return [
        TimeSeries(df[df[RawData.CNPJ_Fundo] == cnpj].reset_index(), cnpj)
        for cnpj in df[RawData.CNPJ_Fundo].unique()
    ]


def load_cdi() -> pd.DataFrame:
    return pd.read_csv(PREPROCESSED_FILES.joinpath("cdi.csv"))


def main():
    funds = load_all_funds()

    timeseries = convert_to_timeseries(funds)
    with open(MODELS_FILES.joinpath("time_series.pkl"), "wb") as file:
        pickle.dump(timeseries, file)

    cdi = load_cdi()
    cdi_timeseries = TimeSeries(cdi, "__CDI__")
    with open(MODELS_FILES.joinpath("cdi_ts.pkl"), "wb") as file:
        pickle.dump(cdi_timeseries, file)


if __name__ == "__main__":
    main()
