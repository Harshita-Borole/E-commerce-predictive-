from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from etl import config as cfg

log = logging.getLogger("etl.extract")


class ExtractionError(RuntimeError):
    pass


def _read_one(name: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{name}.csv"
    if not path.exists():
        raise ExtractionError(
            f"missing source file: {path}\n"
            f"Expected all of {cfg.RAW_TABLES} as .csv in {raw_dir}"
        )

    df = pd.read_csv(path, low_memory=False)

    expected = cfg.EXPECTED_COLUMNS.get(name)
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ExtractionError(
                f"{name}.csv is missing expected column(s): {missing}\n"
                f"found: {list(df.columns)}"
            )

    log.info("extracted %-22s %8s rows x %2d cols", name, f"{len(df):,}", df.shape[1])
    return df


def extract(raw_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    raw_dir = Path(raw_dir) if raw_dir else cfg.RAW_DIR
    log.info("reading raw data from %s", raw_dir)

    tables = {name: _read_one(name, raw_dir) for name in cfg.RAW_TABLES}

    row_counts = {name: len(df) for name, df in tables.items()}

    drift = []
    for name, expected in cfg.BASELINE["raw_rows"].items():
        actual = row_counts.get(name)
        if actual is not None and actual != expected:
            drift.append(f"{name}: expected {expected:,}, found {actual:,}")
            log.warning("row count drift — %s: expected %s, found %s",
                        name, f"{expected:,}", f"{actual:,}")

    stats = {
        "source_dir": (raw_dir.relative_to(cfg.ROOT).as_posix()
                       if raw_dir.is_relative_to(cfg.ROOT) else str(raw_dir)),
        "tables_read": len(tables),
        "row_counts": row_counts,
        "total_rows_extracted": sum(row_counts.values()),
        "row_count_drift": drift,
    }

    log.info("extract complete — %d tables, %s total rows",
             len(tables), f"{stats['total_rows_extracted']:,}")
    return tables, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    _, s = extract()
    print(f"\n{s['total_rows_extracted']:,} rows extracted from {s['tables_read']} tables")
    if s["row_count_drift"]:
        print("WARNING — row counts differ from the EDA baseline:")
        for d in s["row_count_drift"]:
            print("  ", d)
