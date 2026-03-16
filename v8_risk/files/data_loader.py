"""
portfolio_risk/data_loader.py
------------------------------
Responsible for fetching and cleaning all data required by the risk engine.

Exports
-------
DataLoader      : loads and cleans the Everysk portfolio positions.
DICurveLoader   : loads the flat-forward DI curve CSV for a given date.
LoaderResult    : namedtuple bundling both DataFrames for the engine.

All type casting and null handling is centralised here so that processors
always receive DataFrames with guaranteed dtypes and no surprises.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Portfolio positions — column selection
# ---------------------------------------------------------------------------

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

REQUIRED_NON_NULL: list[str] = [
    "port_date",
    "port_link_uid",
    "sec_attribute_ticker",
    "sec_attribute_market_value",
    "sec_attribute_subclasse",
]

# ---------------------------------------------------------------------------
# DI Curve — filesystem config
# ---------------------------------------------------------------------------

DI_CURVE_FOLDER: str = (
    r"\\192.168.0.22\Server_V8\Database\B3\CURVAS"
    r"\PRE_Curva DI X PRÉ\Flat Forward\00_RAW"
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class LoaderResult(NamedTuple):
    """
    Bundles all DataFrames produced by the loading stage.

    Attributes
    ----------
    df_pos : pd.DataFrame
        Clean portfolio positions (COLS, typed, no invalid rows).
    df_di_curve : pd.DataFrame
        DI flat-forward curve with columns [duration_mac, di_rate].
        Empty DataFrame if the curve file is not found for the date.
    """
    df_pos: pd.DataFrame
    df_di_curve: pd.DataFrame


# ---------------------------------------------------------------------------
# DataLoader — Everysk positions
# ---------------------------------------------------------------------------

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

    def load(self) -> LoaderResult:
        """
        Fetch all data needed by the engine for this ref_date.

        Returns
        -------
        LoaderResult
            Named tuple with df_pos and df_di_curve.

        Raises
        ------
        ValueError
            If the position DataFrame is empty after cleaning.
        """
        df_pos = self._load_positions()
        df_di_curve = DICurveLoader(self.ref_date).load()

        return LoaderResult(df_pos=df_pos, df_di_curve=df_di_curve)

    # ------------------------------------------------------------------
    # Private helpers — positions
    # ------------------------------------------------------------------

    def _load_positions(self) -> pd.DataFrame:
        df_raw = self._fetch()
        df = self._select_cols(df_raw)
        df = self._cast_types(df)
        # df = self._drop_invalid_rows(df)

        if df.empty:
            raise ValueError(
                f"No valid rows found for ref_date={self.ref_date}. "
                "Check data source or date."
            )
        return df

    def _fetch(self) -> pd.DataFrame:
        """
        Call EveryskPortfolio with the reference date wrapped in a list.
        Import is deferred to keep this module testable without the dependency.
        """
        from v8_conciliacao.portfolio.everysk import Portfolio as EveryskPortfolio  # noqa: PLC0415

        portfolio_handler = EveryskPortfolio([self.ref_date], list_cnpjs=None)
        return portfolio_handler.df.copy()

    def _select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the columns the engine needs."""
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns from source: {missing}")
        return df[COLS].copy()

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply explicit dtype conversions.

        - port_date        : int YYYYMMDD → datetime64
        - maturity_date    : float YYYYMMDD (nullable) → datetime64 (NaT for nulls)
        - port_nlv         : float
        - market_price     : float
        - quantity         : float
        - market_value     : float
        - look_through_ref : string (CNPJ-like, kept as str to avoid int overflow)
        """
        df = df.copy()

        df["port_date"] = pd.to_datetime(
            df["port_date"], format="%Y%m%d", errors="coerce"
        )
        df["sec_attribute_maturity_date"] = pd.to_datetime(
            df["sec_attribute_maturity_date"],
            format="%Y%m%d",
            errors="coerce",
        )

        numeric_cols = [
            "port_nlv",
            "sec_attribute_market_price",
            "sec_attribute_quantity",
            "sec_attribute_market_value",
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df["sec_attribute_look_through_reference"] = (
            df["sec_attribute_look_through_reference"]
            .dropna()
            .astype('int64')
            .reindex(df.index)
        )

        return df

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows missing critical fields and log them."""
        mask_invalid = df[REQUIRED_NON_NULL].isnull().any(axis=1)
        n_invalid = mask_invalid.sum()

        if n_invalid > 0:
            logger.warning(
                "Dropping %d row(s) with nulls in required columns: %s",
                n_invalid,
                REQUIRED_NON_NULL,
            )

        return df[~mask_invalid].reset_index(drop=True)


# ---------------------------------------------------------------------------
# DICurveLoader — flat-forward DI curve
# ---------------------------------------------------------------------------

class DICurveLoader:
    """
    Loads the flat-forward DI curve CSV for a given reference date.

    Expected file format (semicolon-separated, comma decimal):
        vertice;taxa
        1;0.1050
        ...

    The loader renames columns to [duration_mac, di_rate] and converts
    di_rate from percentage (e.g. 10.50) to decimal (0.1050).

    Parameters
    ----------
    ref_date : date
        Reference date used to locate the CSV file.
    folder : str, optional
        Override the default network folder path.
    """

    def __init__(self, ref_date: date, folder: str = DI_CURVE_FOLDER) -> None:
        self.ref_date = ref_date
        self.folder = folder

    def load(self) -> pd.DataFrame:
        """
        Load and return the DI curve DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: [duration_mac (int), di_rate (float, decimal)].
            Returns an empty DataFrame with the same columns if the file
            is not found — processors must handle this gracefully.
        """
        path = os.path.join(
            self.folder,
            f"{self.ref_date.strftime('%Y%m%d')}_CURVA_PRE.csv",
        )

        if not os.path.exists(path):
            logger.warning(
                "DI curve file not found for ref_date=%s: %s",
                self.ref_date,
                path,
            )
            return pd.DataFrame(columns=["duration_mac", "di_rate"])

        df = pd.read_csv(path, sep=";", decimal=",")
        df.rename(columns={"vertice": "duration_mac", "taxa": "di_rate"}, inplace=True)

        df["di_rate"] = pd.to_numeric(df["di_rate"], errors="coerce") / 100
        df["duration_mac"] = pd.to_numeric(df["duration_mac"], errors="coerce").astype("Int64")
        df.dropna(subset=["duration_mac", "di_rate"], inplace=True)

        logger.debug(
            "DICurveLoader: loaded %d vertices for ref_date=%s", len(df), self.ref_date
        )
        return df.reset_index(drop=True)
