"""
portfolio_risk/data_loader.py
------------------------------
Responsible for fetching and cleaning the raw portfolio DataFrame.

All type casting and null handling is centralised here so that processors
always receive a DataFrame with guaranteed dtypes and no surprises.
"""

from __future__ import annotations
from v8_utilities.logv8 import LogV8

from datetime import date

import pandas as pd

# Lazy import — EveryskPortfolio lives in the caller's project.
# We import inside the method to keep this module independently testable.

logger = LogV8()

# Columns consumed by the risk engine. Any column not listed here is dropped.
COLS: list[str] = [
    "port_date",
    "port_link_uid",
    "port_name",
    "port_nlv",
    "sec_attribute_ticker",
    "sec_attribute_look_through_reference",
    "sec_attribute_trade_id",
    "sec_attribute_market_price",
    "sec_attribute_quantity",
    "sec_attribute_market_value",
    "sec_attribute_maturity_date",
    "sec_attribute_classe",
    "sec_attribute_subclasse",
    "sec_attribute_tipo",
]

# Columns that must exist with non-null values for a row to be valid.
# Rows missing any of these are logged and dropped.
REQUIRED_NON_NULL: list[str] = [
    "port_date",
    "port_link_uid",
    "sec_attribute_ticker",
    "sec_attribute_market_value",
    "sec_attribute_subclasse",
]


class DataLoader:
    """
    Loads the portfolio for a given reference date and returns a clean,
    typed DataFrame ready for the risk processors.

    Parameters
    ----------
    ref_date : date
        The date whose portfolio will be loaded.
    """

    def __init__(self, ref_date: date) -> None:
        self.ref_date = ref_date

    def load(self) -> pd.DataFrame:
        """
        Fetch, select, cast, and validate the raw portfolio data.

        Returns
        -------
        pd.DataFrame
            Clean DataFrame with exactly the columns in COLS, correctly typed.

        Raises
        ------
        ValueError
            If the resulting DataFrame is empty after cleaning.
        """
        df_raw = self._fetch()
        df = self._select_cols(df_raw)
        df = self._drop_invalid_rows(df)

        if df.empty:
            raise ValueError(
                f"No valid rows found for ref_date={self.ref_date}. "
                "Check data source or date."
            )

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch(self) -> pd.DataFrame:
        """
        Call collect_everysk_pos with the reference date wrapped in a list.
        Importing here keeps this module testable without the Everysk dependency.
        """
        from v8_conciliacao.portfolio.everysk import Portfolio as EveryskPortfolio

        portfolio_handler = EveryskPortfolio([self.ref_date], list_cnpjs=None)
        return portfolio_handler.df.copy()

    def _select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns the engine needs."""
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns from source: {missing}")
        return df[COLS].copy()

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows that are missing critical fields and log them."""
        mask_invalid = df[REQUIRED_NON_NULL].isnull().any(axis=1)
        n_invalid = mask_invalid.sum()

        if n_invalid > 0:
            logger.warning(
                "Dropping %d row(s) with nulls in required columns: %s",
                n_invalid,
                REQUIRED_NON_NULL,
            )

        return df[~mask_invalid].reset_index(drop=True)
