from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime

from etl import config as cfg
from etl.extract import extract
from etl.load import (load_to_warehouse, verify_warehouse,
                      write_processed, write_quarantine)
from etl.transform import transform
from etl.validate import validate

BAR = "=" * 78


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, (int, float)) else str(n)


def build_report(ex_stats, tr_stats, val_report, written, rejected, elapsed) -> dict:
    return {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": round(elapsed, 2),
        "extract": ex_stats,
        "transform": {k: v for k, v in tr_stats.items() if k != "features_created"},
        "features_created": tr_stats["features_created"],
        "validation": {
            "total_checks": len(val_report.checks),
            "passed": len(val_report.checks) - len(val_report.failures),
            "failed": len(val_report.failures),
            "critical_failures": [c.name for c in val_report.critical_failures],
            "checks": [{"check": c.name, "result": c.symbol, "detail": c.detail}
                       for c in val_report.checks],
        },
        "load": {"rows_written": written, "rows_rejected": rejected},
    }


def print_report(ex_stats, tr_stats, val_report, written, rejected, elapsed) -> None:
    print(f"\n{BAR}\nETL PIPELINE REPORT\n{BAR}")
    print(f"run at    : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"duration  : {elapsed:.1f}s")
    print(f"source    : {ex_stats['source_dir']}")

    print(f"\n{'-'*78}\nROW RECONCILIATION\n{'-'*78}")
    print(f"{'table':<22}{'extracted':>12}{'transformed':>14}{'delta':>12}   reason")
    print("-" * 78)

    reasons = {
        "geolocation": "collapsed to one row per zip prefix",
        "payments":    "instalment slices aggregated per order",
        "reviews":     "deduplicated to one row per order",
    }
    ex_counts, tr_counts = ex_stats["row_counts"], tr_stats["row_counts"]
    for name in ["customers", "orders", "order_items", "payments",
                 "reviews", "products", "sellers", "geolocation"]:
        e, t = ex_counts.get(name, 0), tr_counts.get(name, 0)
        d = t - e
        print(f"{name:<22}{_fmt(e):>12}{_fmt(t):>14}{_fmt(d):>12}   "
              f"{reasons.get(name, '') if d else ''}")

    print("-" * 78)
    print(f"{'TOTAL EXTRACTED':<22}{_fmt(ex_stats['total_rows_extracted']):>12}")
    print(f"{'TOTAL REJECTED':<22}{_fmt(rejected):>12}"
          f"   {'(see data/processed/quarantine/)' if rejected else '(no rule violations)'}")

    if tr_stats["quarantined"]:
        print("\nrejected rows by rule:")
        for rule, n in tr_stats["quarantined"].items():
            print(f"    {n:>7,}  {rule}")

    print(f"\n{'-'*78}\nDERIVED FEATURE TABLES\n{'-'*78}")
    for name in ["customer_rfm", "seller_performance"]:
        print(f"{name:<22}{_fmt(tr_counts.get(name, 0)):>12} rows")
    print(f"\n{len(tr_stats['features_created'])} engineered features created")

    print(f"\n{'-'*78}\nVALIDATION\n{'-'*78}")
    for c in val_report.checks:
        mark = "PASS" if c.passed else c.severity
        print(f"  [{mark:^8}] {c.name:<46} {c.detail}")

    n_pass = len(val_report.checks) - len(val_report.failures)
    print(f"\n{n_pass}/{len(val_report.checks)} checks passed")

    print(f"\n{'-'*78}\nLOADED TO data/processed/\n{'-'*78}")
    total_loaded = 0
    for name, n in (written or {}).items():
        print(f"{name:<22}{_fmt(n):>12} rows")
        total_loaded += n
    print("-" * 78)
    print(f"{'TOTAL LOADED':<22}{_fmt(total_loaded):>12} rows")

    status = "SUCCESS" if val_report.passed else "FAILED"
    print(f"\n{BAR}\nPIPELINE {status}\n{BAR}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Olist ETL pipeline")
    ap.add_argument("--no-write", action="store_true",
                    help="validate only; do not write data/processed/")
    ap.add_argument("--quiet", action="store_true", help="suppress step logging")
    ap.add_argument("--warehouse", action="store_true",
                    help="also build and load the MySQL star schema")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)-8s %(name)-14s %(message)s",
    )

    t0 = time.perf_counter()

    print(f"\n{BAR}\nEXTRACT\n{BAR}")
    raw, ex_stats = extract()

    print(f"\n{BAR}\nTRANSFORM\n{BAR}")
    tables, tr_stats, quarantine = transform(raw)

    print(f"\n{BAR}\nVALIDATE\n{BAR}")
    report = validate(tables)

    written, rejected = {}, 0
    if args.no_write:
        print("\n--no-write: skipping load")
    elif not report.passed:
        print("\ncritical validation failure — refusing to load")
    else:
        print(f"\n{BAR}\nLOAD\n{BAR}")
        written = write_processed(tables)
        rejected = write_quarantine(quarantine)

    warehouse_loaded, warehouse_checks = {}, None
    if args.warehouse:
        if not report.passed:
            print("\ncritical validation failure — refusing to load the warehouse")
        else:
            print(f"\n{BAR}\nWAREHOUSE (MySQL star schema)\n{BAR}")
            warehouse_loaded = load_to_warehouse(tables)
            warehouse_checks = verify_warehouse()

    elapsed = time.perf_counter() - t0
    print_report(ex_stats, tr_stats, report, written, rejected, elapsed)

    if warehouse_loaded:
        print(f"{'-'*78}\nWAREHOUSE TABLES LOADED\n{'-'*78}")
        for name, n in warehouse_loaded.items():
            print(f"{name:<26}{_fmt(n):>12} rows")
        print(f"\n{'-'*78}\nWAREHOUSE VERIFICATION (queried back out of MySQL)\n{'-'*78}")
        print(warehouse_checks.to_string(index=False))
        n_fail = int((warehouse_checks.result == "FAIL").sum())
        print(f"\n{len(warehouse_checks) - n_fail}/{len(warehouse_checks)} warehouse checks passed\n")

    payload = build_report(ex_stats, tr_stats, report, written, rejected, elapsed)
    out = cfg.REPORT_DIR / "etl_report.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"report saved -> {out}")

    if not report.passed:
        report.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
