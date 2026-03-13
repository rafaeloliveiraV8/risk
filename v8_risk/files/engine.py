"""
portfolio_risk/engine.py
------------------------
Orchestrates the full portfolio risk calculation pipeline.

Flow:
    PortfolioRiskEngine(ref_date)
        └── DataLoader          → raw DataFrame (typed, clean)
        └── SubclasseDispatcher → routes slices to processors (parallel)
            ├── TituloPublicoProcessor
            ├── TituloPrivadoProcessor
            ├── TermoProcessor
            ├── OpcoesProcessor
            ├── ETFProcessor
            └── DefaultProcessor  (fallback for unmapped subclasses)
        └── ResultConsolidator  → final consolidated DataFrame
"""

from __future__ import annotations

from v8_utilities.logv8 import LogV8
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from v8_risk.files.data_loader import DataLoader
from v8_risk.files.processors import (
    DefaultProcessor,
    ETFProcessor,
    OpcoesProcessor,
    TermoProcessor,
    TituloPrivadoProcessor,
    TituloPublicoProcessor,
)
from v8_risk.files.consolidator import ResultConsolidator

logger = LogV8()

# Maps each known sec_attribute_subclasse value to its processor class.
# Add new subclasses here as the coverage expands.
SUBCLASSE_PROCESSOR_MAP: dict[str, type] = {
    "TÍTULO PÚBLICO": TituloPublicoProcessor,
    "TÍTULO PRIVADO": TituloPrivadoProcessor,
    "TERMO": TermoProcessor,
    "OPÇÕES": OpcoesProcessor,
    "ETF": ETFProcessor,
}


class PortfolioRiskEngine:
    """
    Entry point for the portfolio risk calculation pipeline.

    Parameters
    ----------
    ref_date : date
        Reference date for which the portfolio will be loaded and analysed.
    max_workers : int, optional
        Number of threads for parallel processor execution. Defaults to the
        number of distinct subclasses found in the portfolio.
    """

    def __init__(self, ref_date: date, max_workers: int | None = None) -> None:
        self.ref_date = ref_date
        self.max_workers = max_workers
        self._loader = DataLoader(ref_date)
        self._consolidator = ResultConsolidator()

    def run(self) -> pd.DataFrame:
        """
        Execute the full pipeline and return the consolidated risk DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per position, with base columns plus risk metric columns
            appended by each processor.
        """
        logger.info("Pipeline started for ref_date=%s", self.ref_date)

        # 1. Load and clean raw portfolio data
        df_raw = self._loader.load()
        logger.info("DataLoader: %d rows loaded", len(df_raw))

        # 2. Dispatch subclasses to processors in parallel
        dispatcher = SubclasseDispatcher(
            df=df_raw,
            processor_map=SUBCLASSE_PROCESSOR_MAP,
            max_workers=self.max_workers,
        )
        processed_frames = dispatcher.dispatch()
        logger.info("Dispatcher: %d subclass(es) processed", len(processed_frames))

        # 3. Consolidate all processed slices into a single DataFrame
        df_final = self._consolidator.consolidate(processed_frames)
        logger.info("Consolidator: final df shape=%s", df_final.shape)

        return df_final


class SubclasseDispatcher:
    """
    Groups the portfolio DataFrame by sec_attribute_subclasse and dispatches
    each group to the appropriate processor, running them concurrently via
    ThreadPoolExecutor.

    Parameters
    ----------
    df : pd.DataFrame
        Clean portfolio DataFrame produced by DataLoader.
    processor_map : dict[str, type]
        Mapping from subclasse string to processor class.
    max_workers : int | None
        Thread pool size. Defaults to the number of distinct subclasses.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor_map: dict[str, type],
        max_workers: int | None = None,
    ) -> None:
        self._df = df
        self._processor_map = processor_map
        self._max_workers = max_workers

    def dispatch(self) -> list[pd.DataFrame]:
        """
        Run all processors in parallel and collect their output DataFrames.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per subclasse, with risk columns appended.
        """
        groups = {
            subclasse: slice_df
            for subclasse, slice_df in self._df.groupby(
                "sec_attribute_subclasse", sort=False
            )
        }

        n_workers = self._max_workers or len(groups)
        results: list[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._process_group, subclasse, slice_df): subclasse
                for subclasse, slice_df in groups.items()
            }

            for future in as_completed(futures):
                subclasse = futures[future]
                try:
                    results.append(future.result())
                    logger.debug("Processor finished: subclasse='%s'", subclasse)
                except Exception:
                    logger.exception(
                        "Processor failed for subclasse='%s' — skipping", subclasse
                    )

        return results

    def _process_group(self, subclasse: str, df_slice: pd.DataFrame) -> pd.DataFrame:
        """Resolve the correct processor for a subclasse and run it."""
        processor_cls = self._processor_map.get(subclasse, DefaultProcessor)

        if processor_cls is DefaultProcessor and subclasse not in self._processor_map:
            logger.warning(
                "No processor registered for subclasse='%s'. Using DefaultProcessor.",
                subclasse,
            )

        processor = processor_cls()
        return processor.process(df_slice)
