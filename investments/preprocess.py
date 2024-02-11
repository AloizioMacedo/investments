from pathlib import Path

import pandas as pd

DATA_FOLDER = Path("data")

RAW_FILES = DATA_FOLDER.joinpath("01_raw")
FUNDS = RAW_FILES.joinpath("fundos")
PREPROCESSED_FILES = DATA_FOLDER.joinpath("02_preprocessed")


dfs = []


def main():
    for file in FUNDS.iterdir():
        df = pd.read_csv(file, sep=";", encoding="ISO-8859-1")
        dfs.append(df)

    df = pd.concat(dfs)
    df = df[
        [
            "CNPJ_Fundo",
            "Data_Referencia",
            "Percentual_Rentabilidade_Efetiva_Mes",
            "Percentual_Rentabilidade_Patrimonial_Mes",
        ]
    ].sort_values(["CNPJ_Fundo", "Data_Referencia"])

    df["Percentual_Rentabilidade_Efetiva_Mes"] = (
        1 + df["Percentual_Rentabilidade_Efetiva_Mes"]
    )

    df["Percentual_Rentabilidade_Patrimonial_Mes"] = (
        1 + df["Percentual_Rentabilidade_Patrimonial_Mes"]
    )

    df.to_csv(PREPROCESSED_FILES.joinpath("funds.csv"), index=False)


if __name__ == "__main__":
    main()
