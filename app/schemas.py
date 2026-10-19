import re
from datetime import date

from pydantic import BaseModel, Field, validator

HEX32 = re.compile(r"^[0-9a-f]{32}$")
TICKER = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


class TickerRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=15, example="GME")
    start: date
    end: date

    @validator("ticker")
    def _check_ticker(cls, value):
        value = value.strip().upper()
        if not TICKER.match(value):
            raise ValueError(
                "Ticker may contain letters, digits, dots, hyphens, ^ or ="
            )
        return value

    @validator("end")
    def _check_end(cls, value, values):
        if "start" in values and value <= values["start"]:
            raise ValueError("end must be after start")
        return value


class RunRequest(BaseModel):
    file_id: str = Field(..., description="ID returned by the data endpoints")
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    starting_capital: float = Field(10000.0, gt=0, description="Starting cash")
    max_fraction_per_trade: float = Field(
        0.25, gt=0, le=1, description="Largest share of equity used in one position"
    )
    max_shares_per_trade: int = Field(500, gt=0, description="Upper limit on shares per position")
    fee_rate: float = Field(0.001, ge=0, le=0.05, description="Fee as a fraction of trade value")
    window_size: int = Field(10, ge=2, le=60, description="Days of history the agent sees")
    training_steps: int = Field(100000, ge=1000, le=2000000)
    learning_rate: float = Field(0.0007, gt=0, le=0.1)
    allow_short: bool = True
    seed: int = 42

    @validator("file_id")
    def _check_file_id(cls, value):
        if not HEX32.match(value):
            raise ValueError("file_id must be a 32-character lowercase hex string")
        return value