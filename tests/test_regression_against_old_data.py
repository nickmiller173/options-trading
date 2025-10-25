"""Regression tests that compare helper outputs against the legacy dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from datafunctions_AV import _add_return_targets, _compute_days_to_earnings, assign_sector


DATA_ARCHIVE = Path(__file__).resolve().parents[1] / "modelingdf.csv.zip"


def _load_legacy_rows(symbol: str, columns: list[str], limit: int = 8) -> pd.DataFrame:
    """Load up to ``limit`` rows for ``symbol`` from the legacy modelling file."""

    parse_dates = [col for col in ("OpenD", "CloseD") if col in columns]
    collected: list[pd.DataFrame] = []
    rows_needed = limit

    for chunk in pd.read_csv(
        DATA_ARCHIVE,
        usecols=columns,
        chunksize=50_000,
        parse_dates=parse_dates,
    ):
        subset = chunk[chunk["Stock"] == symbol]
        if not subset.empty:
            collected.append(subset)
            rows_needed -= len(subset)
            if rows_needed <= 0:
                break

    if not collected:
        raise RuntimeError(f"Could not find rows for symbol {symbol!r} in legacy data")

    df = pd.concat(collected, ignore_index=True).head(limit)

    numeric_cols = {
        col
        for col in columns
        if col not in {"Stock", "Sector", "OpenD", "CloseD"}
    }
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def legacy_aapl() -> pd.DataFrame:
    columns = [
        "Stock",
        "OpenD",
        "CloseD",
        "OPrice",
        "CPrice",
        "gt10p",
        "gt10n",
        "sev10p",
        "sev10n",
        "gt5p",
        "gt5n",
        "sev5p",
        "sev5n",
        "sev5pcap",
        "chg",
        "pchg",
        "DaysTilEarn",
        "Sector",
    ]
    return _load_legacy_rows("AAPL", columns)


def test_add_return_targets_matches_legacy_values(legacy_aapl: pd.DataFrame) -> None:
    baseline = legacy_aapl[["OPrice", "CPrice"]].copy()
    computed = _add_return_targets(baseline)

    expected_cols = [
        "gt10p",
        "gt10n",
        "sev10p",
        "sev10n",
        "gt5p",
        "gt5n",
        "sev5p",
        "sev5n",
        "sev5pcap",
        "chg",
    ]

    pd.testing.assert_frame_equal(
        computed[expected_cols],
        legacy_aapl[expected_cols],
        check_dtype=False,
        check_like=True,
    )


def test_days_to_earnings_matches_legacy_values(legacy_aapl: pd.DataFrame) -> None:
    open_dates = legacy_aapl[["OpenD"]].copy()
    open_dates["timedif"] = np.arange(len(open_dates))

    earnings_dates = (
        legacy_aapl["OpenD"]
        + pd.to_timedelta(legacy_aapl["DaysTilEarn"].fillna(0), unit="D")
    )

    computed = _compute_days_to_earnings(open_dates, earnings_dates.dropna())

    pd.testing.assert_series_equal(
        computed,
        legacy_aapl["DaysTilEarn"],
        check_names=False,
        check_dtype=False,
    )


def test_assign_sector_matches_legacy_metadata(legacy_aapl: pd.DataFrame) -> None:
    expected_sector = legacy_aapl["Sector"].iloc[0]
    assert assign_sector("AAPL") == expected_sector
