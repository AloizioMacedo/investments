from typing import List

from pydantic import BaseModel


class FundsFilters(BaseModel):
    volatility_threshold: float
    funds: List[str]


class Config(BaseModel):
    funds_filters: FundsFilters
