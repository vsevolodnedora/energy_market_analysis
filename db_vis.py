#!/usr/bin/env python3
"""
Monitor an ENTSO-E Parquet landing-zone while it is being written.

Prints the shape of each logical table:
  cadence=... | freq=... | country_area=... / bidding_zone=...

Optionally saves small matplotlib plots for numeric columns:
  - x-axis spans all rows
  - y-axis is clipped to the central 95% of values
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from collections import defaultdict

import polars as pl

import matplotlib
matplotlib.use("Agg")  # never show plots
import matplotlib.pyplot as plt


def hive_parts(path: Path) -> dict[str, str]:
    out = {}
    for part in path.parts:
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def table_key(path: Path) -> str:
    p = hive_parts(path)
    scope = (
        f"country_area={p['country_area']}"
        if "country_area" in p
        else f"bidding_zone={p.get('bidding_zone', 'UNKNOWN')}"
    )
    return f"cadence={p.get('cadence')} | freq={p.get('freq')} | {scope}"


def discover_tables(root: Path) -> dict[str, list[Path]]:
    files = sorted((root / "data").rglob("data.parquet"))
    tables = defaultdict(list)
    for f in files:
        tables[table_key(f)].append(f)
    return dict(tables)


def scan_files(files: list[Path]) -> pl.LazyFrame:
    scans = [pl.scan_parquet(str(f), hive_partitioning=True) for f in files]
    return scans[0] if len(scans) == 1 else pl.concat(scans, how="diagonal_relaxed")


def retry(fn, *, attempts: int = 3, delay: float = 0.25):
    last_exc = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
    raise last_exc


def table_shape(files: list[Path]) -> tuple[int, int, list[str]]:
    def _read():
        lf = scan_files(files)
        schema = lf.collect_schema()
        columns = schema.names() if hasattr(schema, "names") else list(schema)
        rows = lf.select(pl.len()).collect().item()
        return int(rows), len(columns), list(columns)

    return retry(_read)


def safe_name(text: str) -> str:
    return (
        text.replace(" ", "")
        .replace("|", "__")
        .replace("=", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )


def plot_table(
    key: str,
    files: list[Path],
    out_dir: Path,
    *,
    max_cols: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _collect():
        lf = scan_files(files)
        schema = lf.collect_schema()

        numeric_cols = [
            name
            for name, dtype in schema.items()
            if name not in {
                "date",
                "cadence",
                "freq",
                "country_area",
                "bidding_zone",
                "year",
                "month",
            }
            and dtype.is_numeric()
        ]

        numeric_cols = numeric_cols[:max_cols]
        if not numeric_cols:
            return None, []

        keep = ["date"] if "date" in schema else []
        keep += numeric_cols

        df = (
            lf.select(keep)
            .sort("date") if "date" in keep else lf.select(keep)
        ).collect()

        return df, numeric_cols

    result = retry(_collect)
    if result[0] is None:
        return

    df, numeric_cols = result
    if df.is_empty():
        return

    x = df["date"].to_list() if "date" in df.columns else list(range(df.height))

    for col in numeric_cols:
        s = df[col].drop_nulls()
        if s.is_empty():
            continue

        qlo = s.quantile(0.025)
        qhi = s.quantile(0.975)

        if qlo is None or qhi is None:
            continue

        if qlo == qhi:
            pad = abs(float(qlo)) * 0.01 or 1.0
            qlo, qhi = float(qlo) - pad, float(qhi) + pad

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, df[col].to_list(), linewidth=0.7)

        ax.set_title(f"{key}\n{col}")
        ax.set_xlabel("date" if "date" in df.columns else "row")
        ax.set_ylabel(col)

        ax.set_ylim(float(qlo), float(qhi))
        ax.set_xlim(min(x), max(x))

        fig.autofmt_xdate()
        fig.tight_layout()

        filename = f"{safe_name(key)}__{safe_name(col)}.png"
        fig.savefig(out_dir / filename, dpi=140)
        plt.close(fig)


def print_snapshot(root: Path, plot_dir: Path | None, max_plot_cols: int) -> None:
    tables = discover_tables(root)

    print(f"\nSnapshot: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Root: {root}")
    print(f"Tables: {len(tables)}")

    if not tables:
        print("No parquet tables found.")
        return

    for key, files in sorted(tables.items()):
        try:
            rows, cols, columns = table_shape(files)
            print(f"- {key}: shape=({rows:,}, {cols}) files={len(files)}")
            print(f"  columns: {', '.join(columns[:12])}" + (" ..." if len(columns) > 12 else ""))

            if plot_dir is not None:
                plot_table(key, files, plot_dir, max_cols=max_plot_cols)

        except Exception as exc:
            print(f"- {key}: ERROR while reading active table: {type(exc).__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./entsoe_database", help="Path to EntsoeParquetStore root")
    ap.add_argument("--interval", type=float, default=5.0, help="Seconds between snapshots")
    ap.add_argument("--once", action="store_true", help="Run one snapshot and exit")
    ap.add_argument("--plot-dir", default="./db_plots", help="If set, save matplotlib PNGs here")
    ap.add_argument("--max-plot-cols", type=int, default=4, help="Max numeric columns plotted per table")
    args = ap.parse_args()

    root = Path(args.root)
    plot_dir = Path(args.plot_dir) if args.plot_dir else None

    while True:
        print_snapshot(root, plot_dir, args.max_plot_cols)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()