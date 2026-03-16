"""
portfolio_risk/consolidator.py
-------------------------------
Merges all processor outputs into a single validated DataFrame.
"""

from __future__ import annotations

import logging

import pandas as pd

from v8_risk.files.processors import RISK_COLUMNS

logger = logging.getLogger(__name__)

# Minimum columns that must exist in the final DataFrame.
# Processors guarantee RISK_COLUMNS; base columns come from DataLoader.
EXPECTED_BASE_COLS: list[str] = [
    "port_date",
    "port_link_uid",
    "port_name",
    "port_nlv",
    "sec_attribute_ticker",
    "sec_attribute_market_value",
    "sec_attribute_subclasse",
]


class ResultConsolidator:
    """
    Concatenates the list of processed DataFrames produced by the dispatcher,
    validates the schema, and returns a clean, sorted final DataFrame.
    """

    def consolidate(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all frames, validate schema, and sort for reproducibility.

        Parameters
        ----------
        frames : list[pd.DataFrame]
            One DataFrame per subclasse, as returned by each processor.

        Returns
        -------
        pd.DataFrame
            Consolidated DataFrame sorted by (port_link_uid, sec_attribute_subclasse,
            sec_attribute_ticker).

        Raises
        ------
        ValueError
            If `frames` is empty or the concatenated result is empty.
        """
        if not frames:
            raise ValueError(
                "No processed frames to consolidate. "
                "Check dispatcher logs for processor errors."
            )

        df = pd.concat(frames, ignore_index=True)

        if df.empty:
            raise ValueError("Consolidated DataFrame is empty after concat.")

        df = self._validate_schema(df)
        df = self._sort(df)

        logger.info(
            "Consolidation complete: %d rows | %d funds | %d subclasses",
            len(df),
            df["port_link_uid"].nunique(),
            df["sec_attribute_subclasse"].nunique(),
        )

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all expected columns exist. Missing risk columns are added
        as NaN (should not happen if processors are correct, but guards
        against future regressions).
        """
        missing_base = [c for c in EXPECTED_BASE_COLS if c not in df.columns]
        if missing_base:
            raise ValueError(
                f"Consolidated df is missing required base columns: {missing_base}"
            )

        for col, dtype in RISK_COLUMNS.items():
            if col not in df.columns:
                logger.warning(
                    "Risk column '%s' missing after consolidation — filling with NaN.",
                    col,
                )
                df[col] = pd.NA if dtype is str else float("nan")

        return df

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort for deterministic output order."""
        return df.sort_values(
            by=["port_link_uid", "sec_attribute_subclasse", "sec_attribute_ticker"],
            ignore_index=True,
        )
