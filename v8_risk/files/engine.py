"""
portfolio_risk/engine.py
------------------------
Orchestrates the full portfolio risk calculation pipeline.

Flow:
    PortfolioRiskEngine(ref_date)
        └── DataLoader          → LoaderResult(df_pos, df_di_curve)
        └── SubclasseDispatcher → routes slices to processor *instances* (parallel)
            ├── TituloPublicoProcessor()
            ├── TituloPrivadoProcessor()
            ├── TermoProcessor(df_di_curve=df_di_curve)
            ├── OpcoesProcessor()
            ├── ETFProcessor()
            └── DefaultProcessor()  (fallback for unmapped subclasses)
        └── ResultConsolidator  → final consolidated DataFrame

Design note — processor instances vs classes
--------------------------------------------
The SUBCLASSE_PROCESSOR_MAP was changed from {str: type} to
{str: BaseSubclasseProcessor} (already-instantiated objects). This allows
processors that require external data (e.g. TermoProcessor needs df_di_curve)
to receive it cleanly at construction time via __init__, without forcing a
one-size-fits-all signature on the base class or the dispatcher.

The engine builds the map after loading data, so all dependencies are
available at instantiation time.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

from .data_loader import DataLoader
from .processors import (
    BaseSubclasseProcessor,
    DefaultProcessor,
    ETFProcessor,
    OpcoesProcessor,
    TermoProcessor,
    TituloPrivadoProcessor,
    TituloPublicoProcessor,
)
from .consolidator import ResultConsolidator

logger = logging.getLogger(__name__)


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

        # 1. Load positions and all auxiliary data
        loader_result = self._loader.load()
        df_pos = loader_result.df_pos
        df_di_curve = loader_result.df_di_curve
        logger.info("DataLoader: %d position rows loaded", len(df_pos))

        # 2. Build processor map with fully-instantiated processors.
        #    Processors that need auxiliary data receive it here.
        processor_map: dict[str, BaseSubclasseProcessor] = {
            "TÍTULO PÚBLICO": TituloPublicoProcessor(),
            "TÍTULO PRIVADO": TituloPrivadoProcessor(),
            "TERMO":          TermoProcessor(df_di_curve=df_di_curve),
            "OPÇÕES":         OpcoesProcessor(),
            "ETF":            ETFProcessor(),
        }

        # 3. Dispatch subclasses to processors in parallel
        dispatcher = SubclasseDispatcher(
            df=df_pos,
            processor_map=processor_map,
            max_workers=self.max_workers,
        )
        processed_frames = dispatcher.dispatch()
        logger.info("Dispatcher: %d subclass(es) processed", len(processed_frames))

        # 4. Consolidate all processed slices into a single DataFrame
        df_final = self._consolidator.consolidate(processed_frames)
        logger.info("Consolidator: final df shape=%s", df_final.shape)

        return df_final


class SubclasseDispatcher:
    """
    Groups the portfolio DataFrame by sec_attribute_subclasse and dispatches
    each group to the appropriate processor instance, running them concurrently
    via ThreadPoolExecutor.

    Parameters
    ----------
    df : pd.DataFrame
        Clean portfolio DataFrame produced by DataLoader.
    processor_map : dict[str, BaseSubclasseProcessor]
        Mapping from subclasse string to an already-instantiated processor.
    max_workers : int | None
        Thread pool size. Defaults to the number of distinct subclasses.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor_map: dict[str, BaseSubclasseProcessor],
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
        """Resolve the correct processor instance for a subclasse and run it."""
        processor = self._processor_map.get(subclasse)

        if processor is None:
            logger.warning(
                "No processor registered for subclasse='%s'. Using DefaultProcessor.",
                subclasse,
            )
            processor = DefaultProcessor()

        return processor.process(df_slice)
