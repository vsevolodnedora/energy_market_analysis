#!/usr/bin/env python3
"""Create a compact ENTSO-E parquet database presence report.

The script reads the landing-zone layout used by EntsoeParquetStore directly:

    data/cadence={cadence}/freq={freq}/{scope}={value}/year={YYYY}/month={MM}/data.parquet

For daily and sub-daily physical frequencies it prints one marker per day per
quantity column. For lower-frequency or event datasets it prints total non-null
point counts and an approximate expected count when that makes sense.

Marker legend for daily/sub-daily rows:
    0 = no non-null values for the quantity on that UTC day
    1..4 = less than expected, rough completeness bucket
    5 = exactly expected
    6..9 = more than expected, rough over-completeness bucket

Defaults are intentionally dependency-light: this script does not import your
project package, only Polars, so it can be run next to a copied database.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Iterable

import polars as pl

# Physical `freq=` labels used by the current store, with a few future-proof
# aliases. The current uploaded interface defines minutely_15, minutely_30,
# hourly, weekly, monthly, yearly, and events.
EXPECTED_PER_UTC_DAY: dict[str, int] = {
    "daily": 1,
    "day": 1,
    "hourly": 24,
    "hour": 24,
    "minutely_60": 24,
    "minutely_30": 48,
    "half_hourly": 48,
    "30min": 48,
    "minutely_15": 96,
    "quarter_hourly": 96,
    "15min": 96,
}

STORE_OWNED_COLUMNS: set[str] = {
    "date",
    "cadence",
    "freq",
    "country_area",
    "bidding_zone",
    "scope",
    "year",
    "month",
    "kind",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a visual data-presence report for an ENTSO-E parquet landing-zone database.",
    )
    parser.add_argument(
        "--database-root",
        default="./entsoe_database",
        help="Path to the ENTSO-E database root, or directly to its data/ directory. Default: ./entsoe_database",
    )
    parser.add_argument(
        "--output",
        default="entsoe_database_presence_report.txt",
        help="Path of the .txt report to write. Default: entsoe_database_presence_report.txt",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional inclusive report start date, e.g. 2024-01-01. Defaults to the first observed daily/sub-daily date.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional inclusive report end date, e.g. 2025-12-31. Defaults to the last observed daily/sub-daily date.",
    )
    parser.add_argument(
        "--full-years",
        action="store_true",
        help="Print full Jan-Dec brackets for any touched year, padding outside the requested/observed span with zeroes.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Only report numeric/boolean quantity columns. By default every non-store-owned column is counted.",
    )
    parser.add_argument(
        "--no-month-separators",
        action="store_true",
        help="Do not insert | separators between months inside yearly brackets.",
    )
    parser.add_argument(
        "--max-warnings",
        type=int,
        default=200,
        help="Maximum file/column warnings to include in the report. Default: 200.",
    )
    return parser.parse_args(argv)


def parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date()


def resolve_data_dir(database_root: Path) -> Path:
    root = database_root.expanduser().resolve()
    if root.name == "data" and root.is_dir():
        return root
    data_dir = root / "data"
    if data_dir.is_dir():
        return data_dir
    raise SystemExit(
        f"Could not find ENTSO-E data directory. Tried {data_dir} and {root} itself."
    )


def hive_parts(path: Path) -> dict[str, str]:
    """Extract key=value path segments from a hive-partitioned parquet path."""
    parts: dict[str, str] = {}
    for segment in path.parts:
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        if key:
            parts[key] = value
    return parts


def db_key_from_parts(parts: dict[str, str]) -> str:
    cadence = parts.get("cadence", "?")
    freq = parts.get("freq", "?")
    if "country_area" in parts:
        scope = f"country_area={parts['country_area']}"
    elif "bidding_zone" in parts:
        scope = f"bidding_zone={parts['bidding_zone']}"
    else:
        scope = "scope=?"
    return f"{cadence}|{freq}|{scope}"


def is_numeric_or_bool(dtype: pl.DataType) -> bool:
    try:
        return bool(dtype.is_numeric() or dtype == pl.Boolean)
    except AttributeError:
        return dtype == pl.Boolean


def quantity_columns(schema: dict[str, pl.DataType], *, numeric_only: bool) -> list[str]:
    cols: list[str] = []
    for name, dtype in schema.items():
        if name in STORE_OWNED_COLUMNS or name.startswith("__"):
            continue
        if numeric_only and not is_numeric_or_bool(dtype):
            continue
        cols.append(name)
    return cols


def marker_for_count(n_values: int, expected: int) -> str:
    """Map a daily count to the requested 0..9 visual marker."""
    if n_values <= 0:
        return "0"
    if expected <= 0:
        return "?"
    if n_values == expected:
        return "5"
    if n_values < expected:
        # Four buckets below exact completeness: 1, 2, 3, 4.
        return str(max(1, min(4, math.ceil(4.0 * n_values / expected))))

    # Four buckets above exact completeness: 6, 7, 8, 9. Cap at 2x expected.
    excess_ratio = min(1.0, (n_values - expected) / expected)
    return str(max(6, min(9, 5 + math.ceil(4.0 * excess_ratio))))


def daterange_inclusive(start: date, end: date) -> Iterable[date]:
    ordinal = start.toordinal()
    end_ordinal = end.toordinal()
    while ordinal <= end_ordinal:
        yield date.fromordinal(ordinal)
        ordinal += 1


def normalize_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date()
    # Polars may return Python date/datetime already. Fall back to ISO parsing.
    try:
        return parse_date_arg(str(value))
    except Exception:
        return None


@dataclass
class QuantityDailyPresence:
    db_key: str
    quantity_name: str
    freq: str
    counts_by_day: DefaultDict[date, int] = field(default_factory=lambda: defaultdict(int))
    min_day: date | None = None
    max_day: date | None = None

    def add_count(self, day: date, count: int) -> None:
        self.counts_by_day[day] += int(count)
        if self.min_day is None or day < self.min_day:
            self.min_day = day
        if self.max_day is None or day > self.max_day:
            self.max_day = day


@dataclass
class QuantityPointSummary:
    db_key: str
    quantity_name: str
    freq: str
    n_points: int = 0
    min_day: date | None = None
    max_day: date | None = None

    def add(self, n_points: int, min_value: Any, max_value: Any) -> None:
        self.n_points += int(n_points or 0)
        min_day = normalize_date(min_value)
        max_day = normalize_date(max_value)
        if min_day is not None and (self.min_day is None or min_day < self.min_day):
            self.min_day = min_day
        if max_day is not None and (self.max_day is None or max_day > self.max_day):
            self.max_day = max_day


def scan_one_file(
    path: Path,
    *,
    numeric_only: bool,
    daily: dict[tuple[str, str, str], QuantityDailyPresence],
    other: dict[tuple[str, str, str], QuantityPointSummary],
    warnings: list[str],
) -> None:
    parts = hive_parts(path)
    freq = parts.get("freq", "")
    db_key = db_key_from_parts(parts)

    try:
        lf = pl.scan_parquet(str(path))
        schema = dict(lf.collect_schema())
    except Exception as exc:  # noqa: BLE001 - diagnostics should continue on other partitions
        warnings.append(f"Could not read schema for {path}: {type(exc).__name__}: {exc}")
        return

    if "date" not in schema:
        warnings.append(f"Skipping {path}: no 'date' column in parquet schema.")
        return

    qcols = quantity_columns(schema, numeric_only=numeric_only)
    if not qcols:
        warnings.append(f"Skipping {path}: no reportable quantity columns found.")
        return

    # Cast defensively; the store writes UTC datetimes, but this keeps the
    # diagnostic robust to copied/handwritten test files.
    date_expr = pl.col("date").cast(pl.Datetime("us", "UTC"), strict=False)

    if freq in EXPECTED_PER_UTC_DAY:
        try:
            grouped = (
                lf.select(
                    [date_expr.dt.date().alias("__day")]
                    + [pl.col(c).is_not_null().cast(pl.Int64).alias(c) for c in qcols]
                )
                .filter(pl.col("__day").is_not_null())
                .group_by("__day")
                .agg([pl.col(c).sum().alias(c) for c in qcols])
                .collect()
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not collect daily counts for {path}: {type(exc).__name__}: {exc}")
            return

        for row in grouped.iter_rows(named=True):
            day = row.get("__day")
            if day is None:
                continue
            for col in qcols:
                count = row.get(col) or 0
                key = (db_key, col, freq)
                if key not in daily:
                    daily[key] = QuantityDailyPresence(db_key=db_key, quantity_name=col, freq=freq)
                daily[key].add_count(day, int(count))
        return

    # Other frequencies / events: summarize total non-null points. Expected
    # counts are computed later from the observed or user-requested date span.
    try:
        row = (
            lf.select(
                [
                    date_expr.min().alias("__min_date"),
                    date_expr.max().alias("__max_date"),
                ]
                + [pl.col(c).is_not_null().sum().alias(c) for c in qcols]
            )
            .collect()
            .row(0, named=True)
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not collect point summary for {path}: {type(exc).__name__}: {exc}")
        return

    for col in qcols:
        key = (db_key, col, freq)
        if key not in other:
            other[key] = QuantityPointSummary(db_key=db_key, quantity_name=col, freq=freq)
        other[key].add(row.get(col) or 0, row.get("__min_date"), row.get("__max_date"))


def expected_other_points(freq: str, start_day: date | None, end_day: date | None) -> str:
    if start_day is None or end_day is None:
        return "N/A"
    if end_day < start_day:
        return "0"
    if freq in EXPECTED_PER_UTC_DAY:
        return str(((end_day - start_day).days + 1) * EXPECTED_PER_UTC_DAY[freq])
    if freq == "weekly":
        return str(((end_day - start_day).days // 7) + 1)
    if freq == "monthly":
        return str((end_day.year - start_day.year) * 12 + (end_day.month - start_day.month) + 1)
    if freq == "yearly":
        return str(end_day.year - start_day.year + 1)
    if freq == "events":
        return "N/A"
    return "N/A"


def expected_subdaily_points(freq: str) -> int:
    try:
        return EXPECTED_PER_UTC_DAY[freq]
    except KeyError:
        raise KeyError(f"No expected daily count configured for freq={freq!r}") from None


def year_bounds_for_output(year: int, start_day: date, end_day: date, *, full_years: bool) -> tuple[date, date]:
    y_start = date(year, 1, 1)
    y_end = date(year, 12, 31)
    if full_years:
        return y_start, y_end
    return max(start_day, y_start), min(end_day, y_end)


def format_year_bracket(
    presence: QuantityDailyPresence,
    *,
    year: int,
    start_day: date,
    end_day: date,
    full_years: bool,
    month_separators: bool,
) -> str:
    expected = expected_subdaily_points(presence.freq)
    left, right = year_bounds_for_output(year, start_day, end_day, full_years=full_years)
    tokens: list[str] = []
    prev_month: int | None = None
    for day in daterange_inclusive(left, right):
        if month_separators and prev_month is not None and day.month != prev_month:
            tokens.append("|")
        tokens.append(marker_for_count(presence.counts_by_day.get(day, 0), expected))
        prev_month = day.month
    return f"{year}:[ {' '.join(tokens)} ]"


def format_daily_line(
    presence: QuantityDailyPresence,
    *,
    start_day: date,
    end_day: date,
    full_years: bool,
    month_separators: bool,
) -> str:
    parts = []
    for year in range(start_day.year, end_day.year + 1):
        parts.append(
            format_year_bracket(
                presence,
                year=year,
                start_day=start_day,
                end_day=end_day,
                full_years=full_years,
                month_separators=month_separators,
            )
        )
    return f"{' '.join(parts)} for {presence.db_key}  |  {presence.quantity_name}"


def choose_report_span(
    daily: dict[tuple[str, str, str], QuantityDailyPresence],
    start_arg: date | None,
    end_arg: date | None,
) -> tuple[date | None, date | None]:
    observed_min = min((p.min_day for p in daily.values() if p.min_day is not None), default=None)
    observed_max = max((p.max_day for p in daily.values() if p.max_day is not None), default=None)
    start_day = start_arg or observed_min
    end_day = end_arg or observed_max
    return start_day, end_day


def write_report(
    output: Path,
    *,
    database_root: Path,
    data_dir: Path,
    daily: dict[tuple[str, str, str], QuantityDailyPresence],
    other: dict[tuple[str, str, str], QuantityPointSummary],
    warnings: list[str],
    start_day: date | None,
    end_day: date | None,
    full_years: bool,
    month_separators: bool,
    max_warnings: int,
    explicit_start_day: date | None = None,
    explicit_end_day: date | None = None,
) -> None:
    lines: list[str] = []
    lines.append("ENTSO-E parquet database presence report")
    lines.append(f"database_root={database_root}")
    lines.append(f"data_dir={data_dir}")
    if start_day is not None and end_day is not None:
        lines.append(f"daily_marker_span={start_day.isoformat()}..{end_day.isoformat()} inclusive")
    else:
        lines.append("daily_marker_span=N/A (no daily/sub-daily data found)")
    lines.append(
        "marker_legend: 0=no values, 1..4=below expected, 5=expected, 6..9=above expected; "
        "expected per UTC day: daily=1, hourly=24, minutely_30=48, minutely_15=96"
    )
    if month_separators:
        lines.append("month_separator: | inside each yearly bracket")
    lines.append("")

    lines.append("<<< DAILY AND SUB-DAILY DATA >>>")
    if not daily:
        lines.append("No daily/sub-daily datasets found.")
    elif start_day is None or end_day is None:
        lines.append("No reportable date span found for daily/sub-daily datasets.")
    elif end_day < start_day:
        lines.append(f"Invalid date span: start={start_day}, end={end_day}")
    else:
        for key in sorted(daily):
            presence = daily[key]
            lines.append(
                format_daily_line(
                    presence,
                    start_day=start_day,
                    end_day=end_day,
                    full_years=full_years,
                    month_separators=month_separators,
                )
            )

    lines.append("")
    lines.append("<<< OTHER DATA >>>")
    if not other:
        lines.append("No lower-frequency/event datasets found.")
    else:
        for key in sorted(other):
            summary = other[key]
            expected_start = explicit_start_day if explicit_start_day is not None else summary.min_day
            expected_end = explicit_end_day if explicit_end_day is not None else summary.max_day
            n_expected = expected_other_points(summary.freq, expected_start, expected_end)
            span = "N/A"
            if summary.min_day is not None and summary.max_day is not None:
                span = f"{summary.min_day.isoformat()}..{summary.max_day.isoformat()}"
            lines.append(
                f"N_points={summary.n_points} (N_expected={n_expected}; observed_span={span}) "
                f"for {summary.db_key}  |  {summary.quantity_name}"
            )

    lines.append("")
    lines.append("<<< SUMMARY >>>")
    lines.append(f"daily_subdaily_quantity_lines={len(daily)}")
    lines.append(f"other_quantity_lines={len(other)}")
    lines.append(f"warnings_total={len(warnings)}")

    if warnings:
        lines.append("")
        lines.append("<<< WARNINGS >>>")
        for item in warnings[:max_warnings]:
            lines.append(item)
        if len(warnings) > max_warnings:
            lines.append(f"... {len(warnings) - max_warnings} more warnings omitted")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    database_root = Path(args.database_root).expanduser().resolve()
    data_dir = resolve_data_dir(database_root)
    output = Path(args.output).expanduser().resolve()
    start_arg = parse_date_arg(args.start)
    end_arg = parse_date_arg(args.end)
    if start_arg is not None and end_arg is not None and end_arg < start_arg:
        raise SystemExit(f"--end ({end_arg}) must be >= --start ({start_arg})")

    files = sorted(data_dir.rglob("data.parquet"))
    if not files:
        raise SystemExit(f"No data.parquet files found under {data_dir}")

    daily: dict[tuple[str, str, str], QuantityDailyPresence] = {}
    other: dict[tuple[str, str, str], QuantityPointSummary] = {}
    warnings: list[str] = []

    for path in files:
        scan_one_file(
            path,
            numeric_only=bool(args.numeric_only),
            daily=daily,
            other=other,
            warnings=warnings,
        )

    start_day, end_day = choose_report_span(daily, start_arg, end_arg)
    write_report(
        output,
        database_root=database_root,
        data_dir=data_dir,
        daily=daily,
        other=other,
        warnings=warnings,
        start_day=start_day,
        end_day=end_day,
        full_years=bool(args.full_years),
        month_separators=not bool(args.no_month_separators),
        max_warnings=max(0, int(args.max_warnings)),
        explicit_start_day=start_arg,
        explicit_end_day=end_arg,
    )
    print(f"Wrote {output}")
    print(f"Daily/sub-daily quantity lines: {len(daily)}")
    print(f"Other quantity lines: {len(other)}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
