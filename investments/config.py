from pathlib import Path
from typing import List

import toml
from pydantic import BaseModel


class FundsFilters(BaseModel):
    volatility_threshold: float
    funds: List[str]


class Portfolio(BaseModel):
    number_of_funds: int
    from_date: str
    to_date: str


class Config(BaseModel):
    funds_filters: FundsFilters
    portfolio: Portfolio


CONFIG_FOLDER = Path("config")
CONFIG_FILE = CONFIG_FOLDER.joinpath("config.toml")

CONFIG = toml.load(CONFIG_FILE)
CONFIG = Config(**CONFIG)
