"""Utility functions for building option datasets from Alpha Vantage data.

The module exposes a small :class:`AlphaVantageClient` helper used to download
historical prices and earnings information and a set of pure functions that
prepare two-week holding-period features for modelling option returns.

How to use this file (quick start)
=================================

1. Create a client so that we can talk to Alpha Vantage::

       from datafunctions_AV import AlphaVantageClient, get_std_ticker, getData
       client = AlphaVantageClient(api_key="YOUR_ALPHA_VANTAGE_KEY")

2. Download the daily bars for the broad market tickers you want to merge into
   each stock's feature set.  The helper stores the result in a dictionary that
   mirrors the structure of the original notebook::

       std_data = {
           "dfSPY": get_std_ticker("SPY", client=client),
           "dfUSO": get_std_ticker("USO", client=client),
           "dfVIXY": get_std_ticker("VIXY", client=client),
           "dfGLD": get_std_ticker("GLD", client=client),
       }

3. Build the modelling table for a particular stock by supplying the symbol,
   the date range that covers the option holding periods you care about, and
   the dictionary created above::

       import pandas as pd
       start = pd.Timestamp("2020-01-01")
       end = pd.Timestamp("2020-12-31")
       dataset = getData("AAPL", start, end, std_data, client=client)

The :func:`getData` function returns a tidy :class:`pandas.DataFrame` with one
row per two-week holding period.  Each row includes the opening/closing prices,
target variables (``sev5p``/``sev5n``), lagged features, variance features,
time-to-earnings values, sector labels, and the merged market indicators.
You can feed the resulting frame directly into scikit-learn, XGBoost, or your
preferred modelling framework.

The goal of the rewrite is to keep the behaviour of the original rough draft
while dramatically improving readability, testability and maintainability.  The
abundant inline comments below intentionally favour clarity over brevity so the
logic is approachable for beginners.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from io import StringIO


ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
DEFAULT_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")

# Two week (10 trading day) holding period expressed in trading days.
HOLDING_PERIOD = 11
WINDOW_SIZE = 14

# Pre-defined sector membership used to derive a categorical feature.
SECTOR_MAP = {
    "Tech": {"MSFT", "META", "AMZN", "NFLX", "NVDA", "GOOGL", "INTC", "TSLA", "ORCL",
              "AMD", "CSCO", "AAPL", "TXN", "MU", "QCOM", "ADBE", "CRM"},
    "Finance": {"GS", "BAC", "JPM", "C", "MS", "BLK", "V", "MA", "AXP", "PGR", "PYPL"},
    "Healthcare": {"LLY", "JNJ", "MRK", "PFE", "GILD", "BMY", "UNH", "MDT"},
    "Energy": {"XOM", "CVX", "COP"},
    "Consumer": {"WMT", "COST", "HD", "KO", "MCD", "NKE", "SBUX", "MDLZ", "MO", "CMG",
                  "CVS", "TGT"},
    "Telecom": {"VZ", "CMCSA", "T"},
}


@dataclass
class AlphaVantageClient:
    """Small wrapper around the Alpha Vantage HTTP API.

    Parameters
    ----------
    api_key:
        Your Alpha Vantage API key.  The environment variable
        ``ALPHAVANTAGE_API_KEY`` is used by default so that you do not need to
        hard-code secrets in notebooks.
    session:
        Optional :class:`requests.Session` object.  Supplying one lets you
        customise retry behaviour or share HTTP connection pools.

    Typical usage
    -------------
    >>> client = AlphaVantageClient()
    >>> spy = client.daily_time_series("SPY")
    >>> earnings = client.earnings_calendar("AAPL")
    """

    api_key: str = DEFAULT_API_KEY
    session: requests.Session = field(default_factory=requests.Session)

    def _request(self, function: str, **params) -> dict:
        """Send a JSON request to Alpha Vantage and return the payload.

        This private helper keeps the error checking in one place and is used by
        :meth:`daily_time_series` and :meth:`earnings_history`.
        """
        request_params = {"function": function, "apikey": self.api_key, **params}
        response = self.session.get(ALPHAVANTAGE_URL, params=request_params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "Error Message" in payload:
            raise ValueError(payload["Error Message"])
        return payload

    def daily_time_series(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Download daily OHLCV bars for ``symbol``.

        Returns
        -------
        pandas.DataFrame
            Columns ``Date``, ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
            The frame is sorted by date and suitable for further feature
            engineering.
        """
        data = self._request("TIME_SERIES_DAILY", symbol=symbol, outputsize=outputsize)
        series = data.get("Time Series (Daily)")
        if not series:
            raise ValueError(f"No time series returned for {symbol}")
        df = (pd.DataFrame.from_dict(series, orient="index", dtype=float)
              .rename(columns=lambda c: c.split(". ")[-1])
              .rename(columns=str.capitalize))
        df.index.name = "Date"
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        return df.sort_values("Date").reset_index(drop=True)

    def earnings_calendar(self, symbol: str) -> pd.DataFrame:
        """Fetch upcoming earnings dates for ``symbol``.

        Alpha Vantage delivers the calendar in CSV format, so we stream the
        response into a :class:`pandas.DataFrame`.
        """
        response = self.session.get(
            ALPHAVANTAGE_URL,
            params={
                "function": "EARNINGS_CALENDAR",
                "symbol": symbol,
                "horizon": "6month",
                "apikey": self.api_key,
                "datatype": "csv",
            },
            timeout=30,
        )
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return df
        df = df.rename(columns={"reportDate": "Earnings Date"})
        df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], errors="coerce")
        return df.dropna(subset=["Earnings Date"])[["Earnings Date"]]

    def earnings_history(self, symbol: str) -> pd.DataFrame:
        """Fetch historical earnings announcements for ``symbol``."""
        data = self._request("EARNINGS", symbol=symbol)
        quarterly = pd.DataFrame(data.get("quarterlyEarnings", []))
        if quarterly.empty:
            return quarterly
        quarterly = quarterly.rename(columns={"reportedDate": "Earnings Date"})
        quarterly["Earnings Date"] = pd.to_datetime(quarterly["Earnings Date"], errors="coerce")
        return quarterly.dropna(subset=["Earnings Date"])[["Earnings Date"]]


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns that turn dates into simple integers.

    ``daten`` is the calendar date encoded as the number of days since the Unix
    epoch, which makes joins across dataframes easier. ``timedif`` is the number
    of days since the start of the input data and acts as a lightweight period
    identifier when we merge market tickers onto the main dataset.
    """
    df = df.copy()
    df["daten"] = (df["Date"].astype("int64") // 10 ** 9) // 86400
    start_date = df["Date"].iloc[0]
    df["timedif"] = (df["Date"] - start_date).dt.days
    return df


def _create_two_week_windows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Construct two-week holding period rows from a daily price frame.

    The helper fills in missing business days, slices the data into chunks of
    ``WINDOW_SIZE`` trading days, then extracts the opening and closing prices
    that define the 10-trading-day holding period.
    """
    df = _add_time_columns(daily_df)
    df = df.set_index("Date").asfreq("B").ffill().reset_index()
    df = _add_time_columns(df)

    df["period"] = np.arange(len(df)) // WINDOW_SIZE
    df["offset"] = np.arange(len(df)) % WINDOW_SIZE

    opens = df[df["offset"] == 0][["period", "Date", "daten", "timedif", "Open"]]
    closes = df[df["offset"] == HOLDING_PERIOD][["period", "Date", "daten", "timedif", "Close"]]

    merged = opens.merge(closes, on="period", suffixes=("_open", "_close"))
    merged = merged.rename(columns={
        "Date_open": "OpenD",
        "Date_close": "CloseD",
        "daten_open": "OpenDn",
        "daten_close": "CloseDn",
        "timedif_open": "td1",
        "timedif_close": "td2",
        "Open": "OPrice",
        "Close": "CPrice",
    })
    merged = merged.sort_values("OpenD").reset_index(drop=True)
    merged["timedif"] = merged["td1"]
    return merged


def _add_return_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary/continuous targets for +/-5% and +/-10% thresholds."""
    df = df.copy()
    df["chg"] = (df["CPrice"] - df["OPrice"]) / df["OPrice"]
    df["pchg"] = df["chg"].shift(1)

    def calc_positive_threshold(change: pd.Series, threshold: float) -> pd.DataFrame:
        """Return hit/severity columns for upside moves beyond ``threshold``."""
        hits = (change > threshold).astype(int)
        severity = np.where(change > threshold, change - threshold, 0.0)
        return pd.DataFrame({"hit": hits, "sev": severity})

    def calc_negative_threshold(change: pd.Series, threshold: float) -> pd.DataFrame:
        """Return hit/severity columns for downside moves beyond ``threshold``."""
        hits = (change < -threshold).astype(int)
        severity = np.where(change < -threshold, -(change + threshold), 0.0)
        return pd.DataFrame({"hit": hits, "sev": severity})

    pos5 = calc_positive_threshold(df["chg"], 0.05)
    neg5 = calc_negative_threshold(df["chg"], 0.05)
    pos10 = calc_positive_threshold(df["chg"], 0.10)
    neg10 = calc_negative_threshold(df["chg"], 0.10)

    df[["gt5p", "sev5p"]] = pos5.values
    df[["gt5n", "sev5n"]] = neg5.values
    df[["gt10p", "sev10p"]] = pos10.values
    df[["gt10n", "sev10n"]] = neg10.values

    for lag in (1, 2, 3):
        # Lagged versions of severity/hit columns capture how recent runs
        # influence the present period.
        df[f"p{lag}sev5p"] = df["sev5p"].shift(lag)
        df[f"p{lag}sev5n"] = df["sev5n"].shift(lag)
        df[f"p{lag}gt5p"] = df["gt5p"].shift(lag)
        df[f"p{lag}gt5n"] = df["gt5n"].shift(lag)

    df["p6wkclaimsp"] = df[["p1gt5p", "p2gt5p", "p3gt5p"]].sum(axis=1)
    df["p6wkclaimsn"] = df[["p1gt5n", "p2gt5n", "p3gt5n"]].sum(axis=1)

    def severity_bins(severity: float) -> int:
        if severity == 0:
            return 0
        if severity <= 0.03:
            return 1
        if severity <= 0.07:
            return 2
        return 3

    df["bsev5p"] = df["sev5p"].apply(severity_bins)
    df["bsev5n"] = df["sev5n"].apply(severity_bins)
    df["sev5pcap"] = df["sev5p"].clip(upper=0.25)
    return df


def _add_variance_features(df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling variance features for price and volume changes."""
    daily = _add_time_columns(daily_df.copy())
    daily["period"] = np.arange(len(daily)) // WINDOW_SIZE
    daily["Volchg"] = daily["Volume"].pct_change()
    daily["Closechg"] = daily["Close"].pct_change()

    variance = (daily.groupby("period")[["Closechg", "Volchg"]].var()
                .rename(columns={"Closechg": "closevar", "Volchg": "volvar"}))

    result = df.merge(variance, left_on="period", right_index=True, how="left")
    result["pclosevar"] = result["closevar"].shift(1)
    result["pvolvar"] = result["volvar"].shift(1)
    result["p2closevar"] = result["closevar"].shift(2)
    result["p2volvar"] = result["volvar"].shift(2)
    result["pclosevarchg"] = result["pclosevar"].pct_change()
    result["pvolvarchg"] = result["pvolvar"].pct_change()
    return result


def _compute_days_to_earnings(df: pd.DataFrame, earnings: pd.Series) -> pd.Series:
    """Return the number of calendar days until the next earnings release."""
    if earnings.empty:
        return pd.Series(np.nan, index=df.index)
    sorted_earnings = np.sort(earnings.values.astype("datetime64[D]"))
    open_dates = df["OpenD"].values.astype("datetime64[D]")
    idx = np.searchsorted(sorted_earnings, open_dates, side="left")
    days = np.where(
        idx < len(sorted_earnings),
        (sorted_earnings[idx] - open_dates).astype("timedelta64[D]").astype(float),
        np.nan,
    )
    return pd.Series(days, index=df.index)


def assign_sector(symbol: str) -> str:
    """Return the human-readable sector for ``symbol`` using ``SECTOR_MAP``."""
    for sector, members in SECTOR_MAP.items():
        if symbol in members:
            return sector
    return "Unknown"


def append_stock(dataset: pd.DataFrame, symbol: str, s_date: pd.Timestamp, e_date: pd.Timestamp,
                 std_tick_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge market ticker features onto ``dataset``.

    Parameters
    ----------
    dataset:
        The existing two-week holding-period dataset returned by :func:`getData`.
    symbol:
        Market ticker symbol (for example ``"SPY"``) to append.
    s_date, e_date:
        Start and end dates for the period of interest.  The helper trims the
        market data to this window before joining.
    std_tick_dict:
        Dictionary that maps keys like ``"dfSPY"`` to pre-loaded daily price
        dataframes.  See the module level example for how to construct it.
    """
    key = f"df{symbol}"
    if key not in std_tick_dict:
        raise KeyError(f"Missing pre-loaded data for {symbol}. Run `get_std_ticker` first.")
    raw = std_tick_dict[key].copy()
    raw = raw[(raw["Date"] >= s_date) & (raw["Date"] <= e_date)].reset_index(drop=True)
    market_windows = _create_two_week_windows(raw)
    market_windows = market_windows.drop(columns=["period"], errors="ignore")
    market_windows = market_windows.add_prefix(f"{symbol}_")
    market_windows = market_windows.rename(columns={f"{symbol}_timedif": "timedif"})
    merged = dataset.merge(market_windows, on="timedif", how="left", suffixes=("", f"_{symbol}"))
    return merged


def get_std_ticker(symbol: str, client: Optional[AlphaVantageClient] = None) -> pd.DataFrame:
    """Convenience helper to download a single market ticker."""
    client = client or AlphaVantageClient()
    return client.daily_time_series(symbol)


def getData(symbol: str, s_date: pd.Timestamp, e_date: pd.Timestamp,
            std_tick_dict: Dict[str, pd.DataFrame],
            client: Optional[AlphaVantageClient] = None) -> pd.DataFrame:
    """Build the full modelling dataset for ``symbol``.

    Parameters
    ----------
    symbol:
        Stock ticker symbol to download (for example ``"AAPL"``).
    s_date, e_date:
        Start and end dates of the analysis window.
    std_tick_dict:
        Dictionary of pre-loaded market tickers, usually built with
        :func:`get_std_ticker`.
    client:
        Optional :class:`AlphaVantageClient`.  A new client is created if you do
        not pass one in.

    Returns
    -------
    pandas.DataFrame
        One row per two-week holding period with all engineered features.  The
        key modelling targets are ``sev5p`` (positive 5% strike severity) and
        ``sev5n`` (negative 5% strike severity).
    """
    client = client or AlphaVantageClient()
    raw_daily = client.daily_time_series(symbol)
    mask = (raw_daily["Date"] >= s_date) & (raw_daily["Date"] <= e_date)
    raw_daily = raw_daily.loc[mask].reset_index(drop=True)

    windows = _create_two_week_windows(raw_daily)
    windows["Stock"] = symbol
    windows["period"] = np.arange(len(windows))

    windows = _add_return_targets(windows)
    windows = _add_variance_features(windows, raw_daily)

    earnings_future = client.earnings_calendar(symbol)
    earnings_history = client.earnings_history(symbol)
    earnings_all = pd.concat([earnings_future, earnings_history], ignore_index=True)
    windows["DaysTilEarn"] = _compute_days_to_earnings(windows, earnings_all["Earnings Date"])

    windows["Sector"] = assign_sector(symbol)

    for market_symbol in ("SPY", "USO", "VIXY", "GLD"):
        windows = append_stock(windows, market_symbol, s_date, e_date, std_tick_dict)

    return windows


def binner(var: pd.Series) -> pd.Series:
    """Convert a numeric series into four bins based on quartiles."""
    quantiles = var.quantile([0.25, 0.5, 0.75])

    def assign(value: float) -> int:
        if value < quantiles[0.25]:
            return 0
        if value < quantiles[0.5]:
            return 1
        if value < quantiles[0.75]:
            return 2
        if value > quantiles[0.75]:
            return 3
        return 0

    return var.apply(assign)


def woe_calc(column: pd.Series, target: pd.Series) -> pd.Series:
    """Calculate the Weight of Evidence (WoE) transformation for ``column``."""
    df = pd.DataFrame({"predictor": column, "target": target})
    overall_mean = df["target"].mean()
    mean_target_per_bin = df.groupby("predictor")["target"].mean()
    woe_values = np.log(mean_target_per_bin / overall_mean)
    return column.map(woe_values)


def calculate_bin_accuracy(df: pd.DataFrame, prediction_col: str, target_col: str) -> Dict[int, float]:
    """Print and return accuracy for each target bin in ``target_col``."""
    bin_accuracy: Dict[int, float] = {}
    for bin_value in sorted(df[target_col].dropna().unique()):
        bin_df = df[df[target_col] == bin_value]
        total = len(bin_df)
        if total == 0:
            bin_accuracy[bin_value] = 0.0
            continue
        correct = (bin_df[prediction_col] == bin_value).sum()
        accuracy = correct / total
        bin_accuracy[bin_value] = accuracy
        print(f"Accuracy for bin {bin_value}: {accuracy:.2f}")
    return bin_accuracy


def plot_relative_mean_target(predictor: str, target: str, df: pd.DataFrame) -> None:
    """Plot the relationship between ``predictor`` and the mean ``target`` value."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    mean_target = df.groupby(predictor)[target].mean().reset_index()
    overall_mean = df[target].mean()
    mean_target["relative_mean_target"] = mean_target[target] / overall_mean

    pct_observations = df[predictor].value_counts(normalize=True).sort_index() * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=pct_observations.index, y=pct_observations.values, color="lightblue", ax=ax1)
    ax1.set_ylabel("Percentage of Observations", color="blue")
    ax1.set_xlabel("Predictor Value")

    ax2 = ax1.twinx()
    sns.lineplot(x=mean_target[predictor], y=mean_target["relative_mean_target"], marker="o",
                 color="red", ax=ax2)
    ax2.set_ylabel("Relative Mean Target Value", color="red")
    ax2.axhline(1, color="black", linestyle="--")
    plt.title(predictor)
    plt.show()


def sevgroups2(sev: float) -> int:
    """Group severity into 4 discrete buckets for quick analysis."""
    if sev == 0:
        return 0
    if sev <= 0.04:
        return 1
    if sev <= 0.08:
        return 2
    return 3
