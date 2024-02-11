from pathlib import Path
from typing import List

import pandas as pd
import toml
from config_schema import Config
from paths import CONFIG_FILE
from raw_data_schema import RawData

DATA_FOLDER = Path("data")

RAW_FILES = DATA_FOLDER.joinpath("01_raw")
FUNDS = RAW_FILES.joinpath("fundos")
PREPROCESSED_FILES = DATA_FOLDER.joinpath("02_preprocessed")

CONFIG = toml.load(CONFIG_FILE)
CONFIG = Config(**CONFIG)


def filter_on_volatility(vol: float, df: pd.DataFrame) -> pd.DataFrame:
    names = df[RawData.CNPJ_Fundo].unique()

    to_keep = []
    for name in names:
        new_df = df[df[RawData.CNPJ_Fundo] == name]
        if new_df[RawData.Percentual_Rentabilidade_Efetiva_Mes].std() <= vol:
            to_keep.append(name)

    return df[df[RawData.CNPJ_Fundo].isin(to_keep)]


def filter_on_names(names: List[str], df: pd.DataFrame) -> pd.DataFrame:
    if not names:
        return df

    return df[df[RawData.CNPJ_Fundo].isin(names)]


def load_all_funds() -> pd.DataFrame:
    dfs = []

    for file in FUNDS.iterdir():
        df = pd.read_csv(file, sep=";", encoding="ISO-8859-1")
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            RawData.CNPJ_Fundo,
            RawData.Data_Referencia,
            RawData.Percentual_Rentabilidade_Efetiva_Mes,
            RawData.Percentual_Rentabilidade_Patrimonial_Mes,
        ]
    ].sort_values([RawData.CNPJ_Fundo, RawData.Data_Referencia])

    df.rename(
        columns={
            RawData.Data_Referencia.value: "dt",
            RawData.Percentual_Rentabilidade_Efetiva_Mes.value: "values",
        },
        inplace=True,
    )

    return df


def main():
    df = load_all_funds()

    df[RawData.Percentual_Rentabilidade_Efetiva_Mes] = (
        1 + df[RawData.Percentual_Rentabilidade_Efetiva_Mes]
    )

    df[RawData.Percentual_Rentabilidade_Patrimonial_Mes] = (
        1 + df[RawData.Percentual_Rentabilidade_Patrimonial_Mes]
    )

    df = filter_on_names(CONFIG.funds_filters.funds, df)
    df = filter_on_volatility(CONFIG.funds_filters.volatility_threshold, df)

    df = trim_columns(df)
    df.to_csv(PREPROCESSED_FILES.joinpath("funds.csv"), index=False)


if __name__ == "__main__":
    main()
