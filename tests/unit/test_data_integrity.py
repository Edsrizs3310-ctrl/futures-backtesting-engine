from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.data_validator import DataValidator
from src.data.ib_fetcher_history import IBFetcherHistoryMixin


class _DummyContract:
    def __init__(self, local_symbol: str) -> None:
        self.localSymbol = local_symbol


class _StubFetcher(IBFetcherHistoryMixin):
    def __init__(self) -> None:
        self._contracts_by_date = {
            datetime(2024, 3, 21): _DummyContract("NEW"),
            datetime(2024, 3, 14): _DummyContract("MID"),
            datetime(2024, 3, 7): _DummyContract("OLD"),
        }
        self._chunks = {
            "NEW": self._make_chunk("2024-03-21", 100.0, 101.0),
            "MID": self._make_chunk("2024-03-14", 90.0, 91.0),
            "OLD": self._make_chunk("2024-03-07", 80.0, 81.0),
        }

    @staticmethod
    def _make_chunk(timestamp: str, open_price: float, close_price: float) -> pd.DataFrame:
        index = pd.DatetimeIndex([pd.Timestamp(timestamp)])
        return pd.DataFrame(
            {
                "open": [open_price],
                "high": [max(open_price, close_price)],
                "low": [min(open_price, close_price)],
                "close": [close_price],
                "average": [(open_price + close_price) / 2.0],
                "volume": [1.0],
            },
            index=index,
        )

    def fetch_chunk(self, contract, current_date, timeframe, duration="1 W") -> pd.DataFrame:  # noqa: ANN001
        return self._chunks[contract.localSymbol].copy()

    def _get_contract_for_date(self, all_contracts, target_date):  # noqa: ANN001
        return self._contracts_by_date.get(target_date)

    def _save_checkpoint(self, symbol, timeframe, last_date, total_fetched):  # noqa: ANN001
        return None


def test_backfill_roll_adjustment_uses_raw_next_chunk_open() -> None:
    """
    Roll adjustment must not double-count the cumulative shift from later rolls.
    """
    fetcher = _StubFetcher()

    result = fetcher._backfill_loop(
        symbol="6E",
        timeframe=type("TF", (), {"file_suffix": "1h"})(),
        start_date=datetime(2024, 3, 21),
        stop_date=datetime(2024, 2, 29),
        all_contracts=[],
        checkpoint_key=None,
    )

    result = result.sort_index()

    assert result.loc[pd.Timestamp("2024-03-14"), "close"] == 100.0
    assert result.loc[pd.Timestamp("2024-03-07"), "close"] == 99.0
    assert result.loc[pd.Timestamp("2024-03-07"), "average"] == 98.5


def test_validator_flags_broken_roll_continuity_and_average_mismatch() -> None:
    """
    Validation should fail when rollover creates a large price jump or raw average drift.
    """
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-06-12 22:00:00"),
            pd.Timestamp("2024-06-13 22:00:00"),
            pd.Timestamp("2024-06-13 23:00:00"),
        ]
    )
    df = pd.DataFrame(
        {
            "open": [2.45835, 1.76840, 1.76855],
            "high": [2.45860, 1.76885, 1.76875],
            "low": [2.45810, 1.76830, 1.76840],
            "close": [2.45850, 1.76870, 1.76860],
            "average": [1.08133, 1.078485, 1.078500],
            "volume": [1000.0, 1000.0, 1000.0],
            "contract": ["6EM4", "6EU4", "6EU4"],
        },
        index=index,
    )

    validator = DataValidator()
    result = validator.validate(df, symbol="6E", timeframe="1h")

    assert result.price_anomalies >= 2
    assert not result.is_valid
    assert any("Average outside candle range" in issue for issue in result.issues)
    assert any("Suspicious roll jumps" in issue for issue in result.issues)
