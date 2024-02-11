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


def load_cdi() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES.joinpath("cdi.csv"))
    df = df.drop(columns=["Acumulado"])
    df = df.melt(id_vars="Ano/Mês", var_name="month", value_name="values")

    MONTHS = {
        "Jan": 1,
        "Fev": 2,
        "Mar": 3,
        "Abr": 4,
        "Mai": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Set": 9,
        "Out": 10,
        "Nov": 11,
        "Dez": 12,
    }

    df["month"] = df["month"].apply(lambda x: str(MONTHS[x]).rjust(2, "0"))
    df = df[df["values"] != "---"]

    df.loc[:, "values"] = df["values"].apply(lambda x: x.replace(",", "."))
    df.loc[:, "values"] = 1 + df["values"].astype(float) / 100
    df.loc[:, "dt"] = df["Ano/Mês"].astype(str) + "-" + df["month"] + "-01"
    df = df.drop(columns=["Ano/Mês", "month"])

    return df


def main():
    funds = load_all_funds()

    funds[RawData.Percentual_Rentabilidade_Efetiva_Mes] = (
        1 + funds[RawData.Percentual_Rentabilidade_Efetiva_Mes]
    )

    funds[RawData.Percentual_Rentabilidade_Patrimonial_Mes] = (
        1 + funds[RawData.Percentual_Rentabilidade_Patrimonial_Mes]
    )

    funds = filter_on_names(CONFIG.funds_filters.funds, funds)
    funds = filter_on_volatility(CONFIG.funds_filters.volatility_threshold, funds)

    funds = trim_columns(funds)
    funds.to_csv(PREPROCESSED_FILES.joinpath("funds.csv"), index=False)

    cdi = load_cdi()
    cdi.to_csv(PREPROCESSED_FILES.joinpath("cdi.csv"), index=False)


if __name__ == "__main__":
    main()
