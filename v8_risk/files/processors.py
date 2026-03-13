"""
portfolio_risk/processors.py
-----------------------------
Abstract base class and concrete processor implementations.

Each processor receives a DataFrame slice for its subclasse, computes
the relevant risk metrics, and returns the slice with new columns appended.

Adding a new processor:
    1. Subclass BaseSubclasseProcessor
    2. Override `process()` and populate the risk columns you need
    3. Register it in SUBCLASSE_PROCESSOR_MAP inside engine.py
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk column schema
# ---------------------------------------------------------------------------
# Define every risk column that processors MAY produce. Columns are added to
# the output as NaN by default and populated by the relevant processor.
# This guarantees a consistent schema across all subclasses in the final df.

RISK_COLUMNS: dict[str, object] = {
    # Duration / rate sensitivity
    "risk_duration_macaulay": float,
    "risk_duration_modified": float,
    "risk_dv01": float,
    # Credit / issuer concentration
    "risk_weight_in_fund": float,       # position MtM / port_nlv
    "risk_weight_in_portfolio": float,  # position MtM / total portfolio NLV
    # Liquidity
    "risk_days_to_maturity": float,
    "risk_bucket_maturity": str,        # "0-30d", "31-60d", "61-90d", "91-180d", ">180d"
    # Return
    "risk_ytd_return": float,
    # Market risk
    "risk_var_95": float,
    "risk_cvar_95": float,
    # Spread
    "risk_credit_spread": float,
}


def _init_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all RISK_COLUMNS to df with NaN / None defaults.
    Called at the start of every processor so callers always get the full schema.
    """
    for col, dtype in RISK_COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.NA if dtype is str else float("nan")
    return df


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseSubclasseProcessor(ABC):
    """
    Contract every concrete processor must fulfil.

    Subclasses override `process()` to compute their specific risk metrics.
    The base class ensures RISK_COLUMNS are always initialised before any
    metric is written, so processors can safely assign without KeyErrors.
    """

    subclasse: str = ""  # informational — set in each concrete class

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public entry point called by the dispatcher.

        Guarantees:
            - Input df is a copy (safe to mutate)
            - All RISK_COLUMNS are pre-initialised with NaN/None
            - _compute() is called to fill the relevant columns
            - Output df has a clean integer index

        Parameters
        ----------
        df : pd.DataFrame
            Slice of the portfolio belonging to this processor's subclasse.

        Returns
        -------
        pd.DataFrame
            Same slice with RISK_COLUMNS appended.
        """
        df = df.copy()
        df = _init_risk_columns(df)
        df = self._compute(df)
        return df.reset_index(drop=True)

    @abstractmethod
    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement the risk metric calculations for this subclasse.
        Receives a df that already has all RISK_COLUMNS initialised.
        Must return the same df with the relevant columns populated.
        """

    def _calc_weight_in_fund(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shared helper: weight of each position within its own fund.
        risk_weight_in_fund = sec_attribute_market_value / port_nlv
        """
        df["risk_weight_in_fund"] = (
            df["sec_attribute_market_value"] / df["port_nlv"]
        )
        return df

    def _calc_days_to_maturity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shared helper: calendar days from port_date to maturity_date.
        Requires both columns to be datetime64.
        """
        has_maturity = df["sec_attribute_maturity_date"].notna()
        df.loc[has_maturity, "risk_days_to_maturity"] = (
            df.loc[has_maturity, "sec_attribute_maturity_date"]
            - df.loc[has_maturity, "port_date"]
        ).dt.days
        return df

    def _calc_maturity_bucket(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shared helper: classify positions into liquidity buckets by
        days_to_maturity. Requires _calc_days_to_maturity to run first.
        """
        days = df["risk_days_to_maturity"]
        conditions = [
            days <= 30,
            days <= 60,
            days <= 90,
            days <= 180,
            days > 180,
        ]
        choices = ["0-30d", "31-60d", "61-90d", "91-180d", ">180d"]
        df["risk_bucket_maturity"] = pd.Series(
            pd.Categorical(
                [
                    choices[next((i for i, c in enumerate(conditions) if c.iloc[j]), 4)]
                    if pd.notna(days.iloc[j])
                    else None
                    for j in range(len(df))
                ],
                categories=choices,
                ordered=True,
            )
        )
        return df


# ---------------------------------------------------------------------------
# Concrete processors
# ---------------------------------------------------------------------------

class TituloPublicoProcessor(BaseSubclasseProcessor):
    """
    Handles: TÍTULO PÚBLICO
    Típicos ativos: LFT, LTN, NTN-B, NTN-F

    Planned metrics:
        - Duration Macaulay / Modified
        - DV01
        - Weight in fund
        - Days to maturity + bucket
        - YTD return
        - VaR / CVaR
    """

    subclasse = "TÍTULO PÚBLICO"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- weight in fund (shared helper already vectorised) ---
        df = self._calc_weight_in_fund(df)

        # --- days to maturity + liquidity bucket ---
        df = self._calc_days_to_maturity(df)
        df = self._calc_maturity_bucket(df)

        # TODO: implement duration, DV01, YTD, VaR, CVaR
        logger.debug(
            "TituloPublicoProcessor: processed %d rows. "
            "Duration/DV01/VaR pending implementation.",
            len(df),
        )
        return df


class TituloPrivadoProcessor(BaseSubclasseProcessor):
    """
    Handles: TÍTULO PRIVADO
    Típicos ativos: DEBENTURE, LF, DPGE, CRI, CRA

    Planned metrics:
        - Duration Macaulay / Modified
        - DV01
        - Credit spread
        - Weight in fund
        - Days to maturity + bucket
        - Issuer concentration (HHI — computed at engine level post-concat)
    """

    subclasse = "TÍTULO PRIVADO"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._calc_weight_in_fund(df)
        df = self._calc_days_to_maturity(df)
        df = self._calc_maturity_bucket(df)

        # TODO: implement duration, DV01, credit spread, VaR, CVaR
        logger.debug(
            "TituloPrivadoProcessor: processed %d rows. "
            "Duration/spread/VaR pending implementation.",
            len(df),
        )
        return df


class TermoProcessor(BaseSubclasseProcessor):
    """
    Handles: TERMO
    Contrato a termo de ações (forward financing sobre ações).

    Planned metrics:
        - Weight in fund
        - Days to maturity + bucket
        - Implied rate / spread
        - VaR / CVaR (equity-driven)
    """

    subclasse = "TERMO"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._calc_weight_in_fund(df)
        df = self._calc_days_to_maturity(df)
        df = self._calc_maturity_bucket(df)

        # TODO: implement implied rate, VaR, CVaR
        logger.debug(
            "TermoProcessor: processed %d rows. "
            "Implied rate/VaR pending implementation.",
            len(df),
        )
        return df


class OpcoesProcessor(BaseSubclasseProcessor):
    """
    Handles: OPÇÕES
    Opções sobre ações ou índices.

    Planned metrics:
        - Delta, Gamma, Vega, Theta (Greeks)
        - Weight in fund (notional vs premium)
        - VaR / CVaR
    """

    subclasse = "OPÇÕES"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._calc_weight_in_fund(df)
        df = self._calc_days_to_maturity(df)

        # TODO: implement Greeks, VaR, CVaR
        logger.debug(
            "OpcoesProcessor: processed %d rows. "
            "Greeks/VaR pending implementation.",
            len(df),
        )
        return df


class ETFProcessor(BaseSubclasseProcessor):
    """
    Handles: ETF
    Exchange-Traded Funds (look-through quando disponível).

    Planned metrics:
        - Weight in fund
        - VaR / CVaR (baseado em série histórica do ETF)
        - Look-through para emissor subjacente quando disponível
    """

    subclasse = "ETF"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._calc_weight_in_fund(df)

        # TODO: implement VaR, CVaR, look-through
        logger.debug(
            "ETFProcessor: processed %d rows. "
            "VaR/look-through pending implementation.",
            len(df),
        )
        return df


class DefaultProcessor(BaseSubclasseProcessor):
    """
    Fallback processor for subclasses not yet mapped.

    Returns the slice unchanged (with RISK_COLUMNS initialised to NaN)
    so no data is silently lost during consolidation.
    """

    subclasse = "DEFAULT"

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.warning(
            "DefaultProcessor: no specific processor for this subclasse. "
            "Risk columns will remain NaN for %d rows.",
            len(df),
        )
        return df
