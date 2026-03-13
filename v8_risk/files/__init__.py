"""
portfolio_risk
==============
Risk calculation engine for multi-fund fixed income portfolios.

Quickstart
----------
    from datetime import date
    from portfolio_risk import PortfolioRiskEngine

    engine = PortfolioRiskEngine(ref_date=date(2026, 3, 11))
    df = engine.run()
"""

from .engine import PortfolioRiskEngine
from .processors import (
    BaseSubclasseProcessor,
    DefaultProcessor,
    ETFProcessor,
    OpcoesProcessor,
    TermoProcessor,
    TituloPrivadoProcessor,
    TituloPublicoProcessor,
    RISK_COLUMNS,
)
from .data_loader import COLS as PORTFOLIO_COLS

__all__ = [
    "PortfolioRiskEngine",
    "BaseSubclasseProcessor",
    "DefaultProcessor",
    "ETFProcessor",
    "OpcoesProcessor",
    "TermoProcessor",
    "TituloPrivadoProcessor",
    "TituloPublicoProcessor",
    "RISK_COLUMNS",
    "PORTFOLIO_COLS",
]
