from pathlib import Path
from typing import List

import pandas as pd

from investments.config import CONFIG
from investments.preprocess.raw_data_schema import RawData

DATA_FOLDER = Path("data")

RAW_FILES = DATA_FOLDER.joinpath("01_raw")
REAL_ESTATE_FUNDS = RAW_FILES.joinpath("fundos_imobiliários")
FUNDS = RAW_FILES.joinpath("fundos")
PREPROCESSED_FILES = DATA_FOLDER.joinpath("02_preprocessed")

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


def filter_on_volatility(vol: float, df: pd.DataFrame) -> pd.DataFrame:
    names = df[RawData.CNPJ_Fundo].unique()

    to_keep = []
    for name in names:
        new_df = df[df[RawData.CNPJ_Fundo] == name]
        if new_df[RawData.Percentual_Rentabilidade_Efetiva_Mes].std() <= vol:
            to_keep.append(name)

    return df[df[RawData.CNPJ_Fundo].isin(to_keep)]


def filter_on_names(df: pd.DataFrame) -> pd.DataFrame:
    names = CONFIG.funds_filters.funds

    if names:
        df = df[df[RawData.CNPJ_Fundo].isin(names)]

    exclude_list = CONFIG.funds_filters.exclude
    if exclude_list:
        df = df[~(df[RawData.CNPJ_Fundo].isin(exclude_list))]

    return df


def load_all_real_estate_funds() -> pd.DataFrame:
    dfs = []

    for file in REAL_ESTATE_FUNDS.iterdir():
        df = pd.read_csv(file, sep=";", encoding="ISO-8859-1")
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs)
    return df


def trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            RawData.CNPJ_Fundo,
            RawData.Data_Referencia,
            RawData.Percentual_Rentabilidade_Efetiva_Mes,
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

    df["month"] = df["month"].apply(lambda x: str(MONTHS[x]).rjust(2, "0"))
    df = df[df["values"] != "---"]

    df.loc[:, "values"] = df["values"].apply(lambda x: x.replace(",", "."))
    df.loc[:, "values"] = 1 + df["values"].astype(float) / 100
    df.loc[:, "dt"] = df["Ano/Mês"].astype(str) + "-" + df["month"] + "-01"
    df = df.drop(columns=["Ano/Mês", "month"])

    return df


def load_xp_funds() -> pd.DataFrame:
    dfs = []

    for file in FUNDS.iterdir():
        df = pd.read_csv(file)
        df = df.drop(columns=["Unnamed: 0", "Acumulado"])

        name = file.name

        name = name.replace("_", "/", 1).split(".csv")[0]
        cnpj, date = name.split("_")

        df = df.T
        df = df.reset_index()
        df.rename(columns={0: "values", "index": "month"}, inplace=True)
        df["month"] = df["month"].apply(lambda x: str(MONTHS[x]).rjust(2, "0"))

        df["dt"] = date + "-" + df["month"] + "-01"
        df["CNPJ_Fundo"] = cnpj
        df = df.drop(columns=["month"])

        df = df[df["values"] != "---"]
        df.loc[:, "values"] = df["values"].apply(lambda x: x.replace(",", "."))
        df.loc[:, "values"] = 1 + df["values"].astype(float) / 100
        dfs.append(df)

    return pd.concat(dfs)


def main():
    real_estate_funds = load_all_real_estate_funds()

    if not real_estate_funds.empty:
        real_estate_funds[RawData.Percentual_Rentabilidade_Efetiva_Mes] = (
            1 + real_estate_funds[RawData.Percentual_Rentabilidade_Efetiva_Mes]
        )

        real_estate_funds[RawData.Percentual_Rentabilidade_Patrimonial_Mes] = (
            1 + real_estate_funds[RawData.Percentual_Rentabilidade_Patrimonial_Mes]
        )
        real_estate_funds = filter_on_names(real_estate_funds)
        real_estate_funds = filter_on_volatility(
            CONFIG.funds_filters.volatility_threshold, real_estate_funds
        )

        real_estate_funds = trim_columns(real_estate_funds)

    funds = load_xp_funds()

    if not funds.empty:
        funds = filter_on_names(funds)

    all_funds = pd.concat([funds, real_estate_funds])
    all_funds.to_csv(PREPROCESSED_FILES.joinpath("funds.csv"), index=False)

    cdi = load_cdi()
    cdi.to_csv(PREPROCESSED_FILES.joinpath("cdi.csv"), index=False)


if __name__ == "__main__":
    main()
