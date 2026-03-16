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
    # Duration / rate sensitivity (generic)
    "risk_duration_macaulay": float,
    "risk_duration_modified": float,
    "risk_dv01": float,
    # Credit / issuer concentration
    "risk_weight_in_portfolio": float,       # position MtM / port_nlv
    # Liquidity
    "risk_days_to_maturity": float,
    "risk_bucket_maturity": str,        # "0-30d", "31-60d", "61-90d", "91-180d", ">180d"
    # Return
    "risk_ytm": float,
    # Market risk
    "risk_var_95": float,
    "risk_cvar_95": float,
    # Spread
    "risk_credit_spread": float,
    
    # ── TERMO-specific ────────────────────────────────────────────────
    "risk_dc": float,                   # calendar days to maturity
    "risk_future_value": float,         # FV = quantity
    "risk_present_value": float,        # PV recalculated from YTM
    "risk_pv_vs_mtm_diff": float,       # PV − sec_attribute_market_value
    "risk_dv01_mod": float,             # DV01 via modified duration
    "risk_di_rate": float,              # DI flat-forward rate at matching vertex
    "risk_pct_cdi": float,              # ytm / di_rate
    "risk_spread_di": float,            # ytm − di_rate
    "risk_carry_1d": float,             # 1-business-day carry
    "risk_carry_21d": float,            # 21-business-day carry
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

    def _calc_weight_in_portfolio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shared helper: weight of each position within its own fund.
        risk_weight_in_portfolio = sec_attribute_market_value / port_nlv
        """
        df["risk_weight_in_portfolio"] = (
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
        df = self._calc_weight_in_portfolio(df)

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
        df = self._calc_weight_in_portfolio(df)
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
    Handles: TERMO — Contratos a termo de ações (equity forward financing).

    Metrics computed
    ----------------
    Risk
        risk_duration_macaulay   : business days from port_date to maturity
        risk_dc                 : calendar days to maturity
        risk_ytm                : implied yield from market price
        risk_duration_modified       : modified duration
        risk_present_value      : PV recalculated from YTM (cross-check vs MtM)
        risk_pv_vs_mtm_diff     : PV − sec_attribute_market_value
        risk_dv01               : symmetric 1bp shock on PV
        risk_weight_in_portfolio     : market_value / port_nlv
        risk_days_to_maturity / risk_bucket_maturity

    Return / carry
        risk_di_rate        : DI flat-forward rate for the matching vertex
        risk_pct_cdi        : ytm / di_rate
        risk_spread_di      : ytm − di_rate
        risk_carry_1d       : expected 1-business-day carry
        risk_carry_21d      : expected 21-business-day carry

    Stress test (parametric + historical via config.yaml)
        stress_pv_{label}   : PV under the shock scenario
        stress_pl_{label}   : P&L under the shock scenario

    Notes
    -----
    - Only rows where look_through_reference == port_link_uid are processed
      (i.e. the fund holds the termo directly, not via a third-party vehicle).
    - Rows with duration_mac <= 0 (expired or bad date) are dropped with a warning.
    - If df_di_curve is empty, DI-dependent columns remain NaN.

    Parameters
    ----------
    df_di_curve : pd.DataFrame
        Flat-forward DI curve loaded by DICurveLoader.
        Columns: [duration_mac (int), di_rate (float, decimal)].
    config_path : str, optional
        Path to config.yaml. Defaults to "config.yaml" in the working directory.
    """

    subclasse = "TERMO"

    def __init__(
        self,
        df_di_curve: pd.DataFrame,
        config_path: str = "config.yaml",
    ) -> None:
        self._df_di_curve = df_di_curve
        self._config = self._load_config(config_path)
        self._calendar = self._load_calendar()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict:
        import yaml  # noqa: PLC0415
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(
                "config.yaml not found at '%s'. Stress test will be skipped.",
                config_path,
            )
            return {}

    @staticmethod
    def _load_calendar():
        from v8_utilities.anbima_calendar import Calendar  # noqa: PLC0415
        return Calendar()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── 1. Calendar arrays ─────────────────────────────────────────
        import numpy as np  # noqa: PLC0415

        holidays = (
            np.array(self._calendar.holidays)
            .astype("datetime64[D]")
            .flatten()
        )
        start_days = df["port_date"].values.astype("datetime64[D]")
        end_days = df["sec_attribute_maturity_date"].values.astype("datetime64[D]")

        # ── 2. Time structure ──────────────────────────────────────────
        df["risk_dc"] = (end_days - start_days).astype("timedelta64[D]").astype(int)
        df["risk_duration_macaulay"] = np.busday_count(
            start_days, end_days, holidays=holidays
        )
        
        # Drop expired / bad-date rows
        mask_invalid = df["risk_duration_macaulay"] < 0
        if mask_invalid.any():
            logger.warning(
                "TermoProcessor: dropping %d row(s) with risk_duration_macaulay < 0.",
                mask_invalid.sum(),
            )
            df = df[~mask_invalid].copy()

        if df.empty:
            return df

        # FV = quantity (the forward price is the PV/FV ratio → FV = shares)
        df["risk_future_value"] = df["sec_attribute_quantity"]

        # ── 3. Risk metrics ────────────────────────────────────────────
        # YTM: price = PV/FV = 1/(1+ytm)^(du/252) → ytm = (1/price)^(252/du) - 1
        df["risk_ytm"] = (
            (1 / df["sec_attribute_market_price"]) ** (252 / df["risk_duration_macaulay"]) - 1
        )
        df["risk_duration_modified"] = (
            (df["risk_duration_macaulay"] / 252) / (1 + df["risk_ytm"])
        )
        df["risk_present_value"] = self._pv(
            df["risk_future_value"], df["risk_ytm"], df["risk_duration_macaulay"]
        )

        # DV01 via symmetric 1bp shock
        pv_up   = self._pv(df["risk_future_value"], df["risk_ytm"] + 0.0001, df["risk_duration_macaulay"])
        pv_down = self._pv(df["risk_future_value"], df["risk_ytm"] - 0.0001, df["risk_duration_macaulay"])
        df["risk_dv01"]     = (pv_down - pv_up) / 2
        df["risk_dv01_mod"] = df["risk_duration_modified"] * df["risk_present_value"] * 0.0001

        df["risk_pv_vs_mtm_diff"] = (
            df["risk_present_value"] - df["sec_attribute_market_value"]
        )

        # Shared base helpers
        df = self._calc_weight_in_portfolio(df)
        df["risk_days_to_maturity"] = df["risk_duration_macaulay"]   # already in biz days
        df = self._calc_maturity_bucket(df)

        # ── 4. Return / carry ─────────────────────────────────────────
        df = self._merge_di_curve(df)

        if "risk_di_rate" in df.columns and df["risk_di_rate"].notna().any():
            df["risk_pct_cdi"]   = df["risk_ytm"] / df["risk_di_rate"]
            df["risk_spread_di"] = df["risk_ytm"] - df["risk_di_rate"]
        else:
            df["risk_pct_cdi"]   = float("nan")
            df["risk_spread_di"] = float("nan")

        df["risk_carry_1d"] = (
            self._pv(df["risk_future_value"], df["risk_ytm"], df["risk_duration_macaulay"] - 1)
            / df["risk_present_value"] - 1
        )
        df["risk_carry_21d"] = (
            self._pv(df["risk_future_value"], df["risk_ytm"], df["risk_duration_macaulay"] - 21)
            / df["risk_present_value"] - 1
        )

        # ── 6. Stress test ────────────────────────────────────────────
        df = self._apply_stress(df)

        logger.debug("TermoProcessor: processed %d rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pv(
        future_value: pd.Series,
        ytm: pd.Series,
        duration_mac: pd.Series,
    ) -> pd.Series:
        """PV = FV / (1 + ytm)^(du/252)"""
        return future_value / (1 + ytm) ** (duration_mac / 252)

    def _merge_di_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Left-join the DI curve on duration_mac to attach di_rate.
        If df_di_curve is empty, di_rate column is added as NaN.
        """
        if self._df_di_curve.empty:
            logger.warning(
                "TermoProcessor: df_di_curve is empty — di_rate will be NaN."
            )
            df["risk_di_rate"] = float("nan")
            return df

        curve = self._df_di_curve.rename(
            columns={"duration_mac": "risk_duration_macaulay", "di_rate": "risk_di_rate"}
        )
        df = df.merge(curve[["risk_duration_macaulay", "risk_di_rate"]], on="risk_duration_macaulay", how="left")
        return df

    def _apply_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply parametric and historical stress scenarios from config.yaml.

        Columns added:
            stress_pv_{label} : PV under the shocked rate
            stress_pl_{label} : P&L vs current present_value
        """
        parametric_shocks = [
            {"name": self._shock_label(bps), "shock_bps": bps}
            for bps in self._config.get("stress_parametric", {}).get("shocks_bps", [])
        ]
        historical_events = self._config.get("stress_historical", {}).get("events", [])

        for event in parametric_shocks + historical_events:
            shock   = event["shock_bps"] / 10_000
            ytm_s   = df["risk_ytm"] + shock
            pv_s    = self._pv(df["risk_future_value"], ytm_s, df["risk_duration_macaulay"])
            label   = event["name"]
            df[f"stress_pv_{label}"] = pv_s
            df[f"stress_pl_{label}"] = pv_s - df["risk_present_value"]

        return df

    @staticmethod
    def _shock_label(bps: int) -> str:
        """Format stress label: +100 → 'p100bps' | -100 → 'm100bps'"""
        return f"p{bps}bps" if bps >= 0 else f"m{abs(bps)}bps"


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
        df = self._calc_weight_in_portfolio(df)
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
        df = self._calc_weight_in_portfolio(df)

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
