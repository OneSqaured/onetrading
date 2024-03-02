import pandas as pd

from onetrading.data import get_tick

import pandas as pd


def fetch_and_prepare_tick_data(symbol, start_date, end_date, tz):
    """
    Fetches raw tick data and prepares it by calculating price_volume.
    """
    tick_df = get_tick(symbol, start_date, end_date, tz)
    tick_df["price_volume"] = tick_df["price"] * tick_df["size"]
    return tick_df


def aggregate_bars(tick_df, bar_size, bar_type):
    """
    Aggregates tick data into bars based on the bar type and size.
    """
    if bar_type == "time":
        tick_df.set_index("ts_event", inplace=True)
        tick_df = (
            tick_df.groupby("symbol")
            .resample(bar_size, label="right")
            .agg(
                {
                    "price": "ohlc",
                    "size": "sum",
                    "price_volume": "sum",
                }
            )
            .reset_index()
        )
    else:
        if bar_type == "tick":
            cumcount = tick_df.groupby("symbol").cumcount()
        elif bar_type == "volume":
            cumcount = tick_df.groupby("symbol")["size"].cumsum()
        elif bar_type == "dollar":
            cumcount = tick_df.groupby("symbol")["price_volume"].cumsum()
        tick_df["bar_num"] = cumcount // bar_size
        tick_df = (
            tick_df.groupby(["symbol", "bar_num"])
            .agg(
                {
                    "ts_event": "last",
                    "price": "ohlc",
                    "size": "sum",
                    "price_volume": "sum",
                }
            )
            .reset_index()
        )

    # Flatten multi-index columns if necessary
    if isinstance(tick_df.columns, pd.MultiIndex):
        tick_df.columns = [
            col[0] if col[1] == "" else col[1] for col in tick_df.columns
        ]

    return tick_df


def finalize_bars(bars_df):
    """
    Finalizes the bars DataFrame by calculating VWAP and selecting relevant columns.
    """
    bars_df["vwap"] = bars_df["price_volume"] / bars_df["size"]
    bars_df = bars_df[
        ["ts_event", "symbol", "vwap", "open", "high", "low", "close", "size"]
    ]
    bars_df = bars_df.sort_values(["symbol", "ts_event"])
    return bars_df


def get_time_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Fetches time bars from raw tick data based on the specified conditions.

    This function queries raw tick data from a database, and aggregates it into
    time bars of the specified frequency. It filters the data by symbol, start
    date, and end date, and adjusts the input dates to UTC for the query. It
    Parameters:
    also converts the resulting timestamps to the desired timezone.

    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".
    - freq (str, optional): The frequency of the time bars. Defaults to "1H".

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered time bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching 1-hour time bars for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_time_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02", freq="1h", tz="US/Eastern")
        print(df.head())

    - Fetching 5-minute time bars for symbol "ESH4" from 2024-02-03 to 2024-02-04, converting timestamps to US/Eastern timezone:
        df = get_time_bar(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern", freq="5min")
        print(df.tail())

    - Fetching all available time bars without filtering:
        df = get_time_bar()
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    bars_df = aggregate_bars(tick_df, freq, "time")
    time_bars = finalize_bars(bars_df)

    return time_bars


def get_tick_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    bar_size: int = 1000,
) -> pd.DataFrame:
    """
    Fetches tick bars from raw tick data based on the specified conditions.

    This function queries raw tick data from a database, and aggregates it into
    tick bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".
    - bar_size (int, optional): The size of the tick bars. Defaults to 1000.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered tick bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching tick bars of size 1000 for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_tick_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02", bar_size=1000)
        print(df.head())

    - Fetching tick bars of size 500 for symbol "ESH4" from 2024-02-03 to 2024-02-04, converting timestamps to US/Eastern timezone:
        df = get_tick_bar(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern", bar_size=500)
        print(df.tail())

    - Fetching all available tick bars without filtering:
        df = get_tick_bar()
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    bars_df = aggregate_bars(tick_df, bar_size, "tick")
    tick_bars = finalize_bars(bars_df)

    return tick_bars


def get_volume_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    bar_size: int = 1000,
) -> pd.DataFrame:
    """
    Fetches volume bars from raw tick data based on the specified conditions.

    This function queries raw tick data from a database, and aggregates it into
    volume bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".
    - bar_size (int, optional): The size of the volume bars. Defaults to 1000.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered volume bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching volume bars of size 1000 for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_volume_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02", bar_size=1000)
        print(df.head())

    - Fetching volume bars of size 500 for symbol "ESH4" from 2024-02-03 to 2024-02-04, converting timestamps to US/Eastern timezone:
        df = get_volume_bar(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern", bar_size=500)
        print(df.tail())

    - Fetching all available volume bars without filtering:
        df = get_volume_bar()
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    bars_df = aggregate_bars(tick_df, bar_size, "volume")
    volume_bars = finalize_bars(bars_df)

    return volume_bars


def get_dollar_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    bar_size: int = 1000000,
) -> pd.DataFrame:
    """
    Fetches dollar bars from raw tick data based on the specified conditions.

    This function queries raw tick data from a database, and aggregates it into
    dollar bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".
    - bar_size (int, optional): The size of the dollar bars. Defaults to 1000000.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered dollar bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching dollar bars of size 1000000 for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_dollar_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02", bar_size=1000000)
        print(df.head())

    - Fetching dollar bars of size 500000 for symbol "ESH4" from 2024-02-03 to 2024-02-04, converting timestamps to US/Eastern timezone:
        df = get_dollar_bar(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern", bar_size=500000)
        print(df.tail())

    - Fetching all available dollar bars without filtering:
        df = get_dollar_bar()
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    bars_df = aggregate_bars(tick_df, bar_size, "dollar")
    dollar_bars = finalize_bars(bars_df)

    return dollar_bars


class Bar:
    def __init__(self, symbol: str, tz: str = "UTC"):
        self.symbol = symbol
        self.tz = tz

    def get_bar(self, start_date: str = None, end_date: str = None):

        raise NotImplementedError


class TimeBar(Bar):

    def get_bar(self, start_date: str = None, end_date: str = None, freq: str = "1min"):
        return get_time_bar([self.symbol], start_date, end_date, self.tz, freq)


class TickBar(Bar):

    def get_bar(
        self, start_date: str = None, end_date: str = None, bar_size: int = 1000
    ):
        return get_tick_bar([self.symbol], start_date, end_date, self.tz, bar_size)


class VolumeBar(Bar):

    def get_bar(
        self, start_date: str = None, end_date: str = None, bar_size: int = 1000
    ):
        return get_volume_bar([self.symbol], start_date, end_date, self.tz, bar_size)


class DollarBar(Bar):

    def get_bar(
        self, start_date: str = None, end_date: str = None, bar_size: int = 1000000
    ):
        return get_dollar_bar([self.symbol], start_date, end_date, self.tz, bar_size)
