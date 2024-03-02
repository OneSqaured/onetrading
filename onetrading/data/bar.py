import numpy as np
import pandas as pd

from onetrading.data import get_tick

import pandas as pd


def flatten_multi_index_columns(df: pd.DataFrame) -> pd.DataFrame:

    # Flatten multi-index columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == "" else col[1] for col in df.columns]

    return df


def preprocess_tick_data(tick_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses raw tick data by calculating dollar_amount and diff_price.
    """
    tick_df["dollar_amount"] = tick_df["price"] * tick_df["size"]
    return tick_df


def fetch_and_prepare_tick_data(
    symbol: list[str], start_date: str, end_date: str, tz: str
) -> pd.DataFrame:
    """
    Fetches raw tick data and prepares it by calculating dollar_amount.
    """
    tick_df = get_tick(symbol, start_date, end_date, tz)
    tick_df["dollar_amount"] = tick_df["price"] * tick_df["size"]
    return tick_df


def aggregate_bars(
    tick_df: pd.DataFrame, threshold: str | int, clock: str
) -> pd.DataFrame:
    """
    Aggregates tick data into bars based on the bar type and size.
    """
    if clock == "time":
        tick_df.set_index("ts_event", inplace=True)
        tick_df = (
            tick_df.groupby("symbol")
            .resample(threshold, label="right")
            .agg(
                {
                    "price": "ohlc",
                    "size": "sum",
                    "dollar_amount": "sum",
                }
            )
            .reset_index()
        )
    else:
        if clock == "tick":
            cumcount = tick_df.groupby("symbol").cumcount()
        elif clock == "volume":
            cumcount = tick_df.groupby("symbol")["size"].cumsum()
        elif clock == "dollar":
            cumcount = tick_df.groupby("symbol")["dollar_amount"].cumsum()
        tick_df["bar_num"] = cumcount // threshold
        tick_df = (
            tick_df.groupby(["symbol", "bar_num"])
            .agg(
                {
                    "ts_event": "last",
                    "price": "ohlc",
                    "size": "sum",
                    "dollar_amount": "sum",
                }
            )
            .reset_index()
        )

    tick_df = flatten_multi_index_columns(tick_df)

    return tick_df


def aggregate_imbalance_bars(
    tick_df: pd.DataFrame,
    initial_size: int,
    alpha: float,
    max_imbalance: float,
    clock: str,
) -> pd.DataFrame:

    tick_df["diff_price"] = tick_df["price"].diff()

    # create b_t
    tick_df["tick_direction"] = np.nan
    tick_df.loc[tick_df["diff_price"] > 0, "tick_direction"] = 1
    tick_df.loc[tick_df["diff_price"] < 0, "tick_direction"] = -1
    tick_df["tick_direction"] = tick_df["tick_direction"].ffill()
    tick_df = tick_df.dropna(ignore_index=True)

    if clock == "tick":
        tick_direction = np.array(tick_df["tick_direction"])
    elif clock == "volume":
        tick_direction = np.array(tick_df["tick_direction"] * tick_df["size"])
    elif clock == "dollar":
        tick_direction = np.array(tick_df["tick_direction"] * tick_df["dollar_amount"])

    sample_idx = np.zeros(len(tick_df))
    sample_idx[initial_size] = 1  # new group at this index
    exp_imbalance = abs(tick_df["tick_direction"][0:initial_size].sum())
    exp_ticks = initial_size

    last_i = initial_size
    i = initial_size + exp_ticks
    imbalance = tick_direction[last_i:i].sum()
    while i < len(sample_idx):
        if abs(imbalance) > exp_imbalance:
            sample_idx[i] = 1
            exp_imbalance = alpha * exp_imbalance + (1 - alpha) * min(
                abs(imbalance), max_imbalance
            )
            exp_ticks = int(alpha * exp_ticks + (1 - alpha) * (i - last_i))
            last_i = i
            i += exp_ticks
            imbalance = tick_direction[last_i:i].sum()
        else:
            imbalance += tick_direction[i]
            i += 1

    pd.options.mode.chained_assignment = None
    tick_df["sample_idx"] = sample_idx
    tick_df["bar_num"] = tick_df["sample_idx"].cumsum()
    pd.options.mode.chained_assignment = "warn"

    tick_df = (
        tick_df.groupby(["symbol", "bar_num"])
        .agg(
            {
                "ts_event": "last",
                "price": "ohlc",
                "size": "sum",
                "dollar_amount": "sum",
            }
        )
        .reset_index()
    )

    tick_df = flatten_multi_index_columns(tick_df)

    return tick_df


def finalize_bars(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finalizes the bars DataFrame by calculating VWAP and selecting relevant columns.
    """
    bars_df["vwap"] = bars_df["dollar_amount"] / bars_df["size"]
    bars_df = bars_df[
        ["ts_event", "symbol", "vwap", "open", "high", "low", "close", "size"]
    ]
    bars_df = bars_df.sort_values(["symbol", "ts_event"])
    return bars_df


def sample_time_bar(tick_df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:

    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_bars(tick_df, freq, "time")
    time_bars = finalize_bars(bars_df)

    return time_bars


def sample_tick_bar(tick_df: pd.DataFrame, bar_size: int = 1000) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_bars(tick_df, bar_size, "tick")
    tick_bars = finalize_bars(bars_df)

    return tick_bars


def sample_volume_bar(tick_df: pd.DataFrame, bar_size: int) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_bars(tick_df, bar_size, "volume")
    volume_bars = finalize_bars(bars_df)

    return volume_bars


def sample_dollar_bar(tick_df: pd.DataFrame, bar_size: int) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_bars(tick_df, bar_size, "dollar")
    dollar_bars = finalize_bars(bars_df)

    return dollar_bars


def sample_tick_imbalance_bar(
    tick_df: pd.DataFrame, initial_size: int, alpha: float, max_imbalance: float
) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_imbalance_bars(
        tick_df, initial_size, alpha, max_imbalance, "tick"
    )
    imbalance_bars = finalize_bars(bars_df)

    return imbalance_bars


def sample_volume_imbalance_bar(
    tick_df: pd.DataFrame, initial_size: int, alpha: float, max_imbalance: float
) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_imbalance_bars(
        tick_df, initial_size, alpha, max_imbalance, "volume"
    )
    imbalance_bars = finalize_bars(bars_df)

    return imbalance_bars


def sample_dollar_imbalance_bar(
    tick_df: pd.DataFrame, initial_size: int, alpha: float, max_imbalance: float
) -> pd.DataFrame:
    tick_df = preprocess_tick_data(tick_df)
    bars_df = aggregate_imbalance_bars(
        tick_df, initial_size, alpha, max_imbalance, "dollar"
    )
    imbalance_bars = finalize_bars(bars_df)

    return imbalance_bars


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

    tick_df = get_tick(symbol, start_date, end_date, tz)
    time_bars = sample_time_bar(tick_df, freq)

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

    tick_df = get_tick(symbol, start_date, end_date, tz)
    tick_bars = sample_tick_bar(tick_df, bar_size)

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

    tick_df = get_tick(symbol, start_date, end_date, tz)
    volume_bars = sample_volume_bar(tick_df, bar_size)

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

    tick_df = get_tick(symbol, start_date, end_date, tz)
    dollar_bars = sample_dollar_bar(tick_df, bar_size)

    return dollar_bars


def get_tick_imbalance_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    initial_size: int = 1000,
    alpha: float = 0.97,
    max_imbalance: float = 100,
) -> pd.DataFrame:
    """
    Fetches tick imbalance bars from raw tick data based on the specified conditions.

    The idea behind tick imbalance bars is to aggregate ticks based on the direction.
    Define theta_T = sum(b_t) where b_t = 1 if tick is uptick, -1 if tick is downtick.
    We would like to sample a bar whenever |theta_T| >= E(theta_T) = E(T)E(2P(b+_t) - 1)
    where E(T) is the expected number of ticks per bar and E(2P(b+_t) - 1) is the expected ratio of the tick imbalance.
    When we samples a tick imbalance bar, it is likely to indicate informed trading is happening in the market.

    This function queries raw tick data from a database, and aggregates it into
    tick imbalance bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered tick imbalance bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching tick imbalance bars for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_tick_imbalance_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02")
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    imbalance_bar = sample_tick_imbalance_bar(
        tick_df, initial_size, alpha, max_imbalance
    )

    return imbalance_bar


def get_volume_imbalance_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    initial_size: int = 1000,
    alpha: float = 0.97,
    max_imbalance: float = 500,
) -> pd.DataFrame:
    """
    Fetches volume imbalance bars from raw tick data based on the specified conditions.

    The idea behind volume imbalance bars is to aggregate ticks based on the direction.
    Define theta_T = sum(b_t) where b_t = 1 if tick is uptick, -1 if tick is downtick.
    We would like to sample a bar whenever |theta_T| >= E(theta_T) = E(T)E(2P(b+_t) - 1)
    where E(T) is the expected number of ticks per bar and E(2P(b+_t) - 1) is the expected ratio of the tick imbalance.
    When we samples a tick imbalance bar, it is likely to indicate informed trading is happening in the market.

    This function queries raw tick data from a database, and aggregates it into
    volume imbalance bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered volume imbalance bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching volume imbalance bars for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_volume_imbalance_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02")
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    imbalance_bar = sample_volume_imbalance_bar(
        tick_df, initial_size, alpha, max_imbalance
    )

    return imbalance_bar


def get_dollar_imbalance_bar(
    symbol: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    tz: str = "UTC",
    initial_size: int = 1000,
    alpha: float = 0.97,
    max_imbalance: float = 1000000,
) -> pd.DataFrame:
    """
    Fetches dollar imbalance bars from raw tick data based on the specified conditions.

    The idea behind dollar imbalance bars is to aggregate ticks based on the direction.
    Define theta_T = sum(b_t) where b_t = 1 if tick is uptick, -1 if tick is downtick.
    We would like to sample a bar whenever |theta_T| >= E(theta_T) = E(T)E(2P(b+_t) - 1)
    where E(T) is the expected number of ticks per bar and E(2P(b+_t) - 1) is the expected ratio of the tick imbalance.
    When we samples a tick imbalance bar, it is likely to indicate informed trading is happening in the market.

    This function queries raw tick data from a database, and aggregates it into
    dollar imbalance bars of the specified size. It filters the data by symbol, start date,
    and end date, and adjusts the input dates to UTC for the query. It also
    converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered dollar imbalance bars, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching dollar imbalance bars for symbol "ESH4" from 2024-02-01 to 2024-02-02:
        df = get_dollar_imbalance_bar(symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02")
        print(df.head())
    """

    tick_df = fetch_and_prepare_tick_data(symbol, start_date, end_date, tz)
    imbalance_bar = sample_dollar_imbalance_bar(
        tick_df, initial_size, alpha, max_imbalance
    )

    return imbalance_bar


class Bar:
    def __init__(self, symbol: str, tz: str = "UTC"):
        self.symbol = symbol
        self.tz = tz

    def __str__(self):
        return f"{self.__class__.__name__}(symbol={self.symbol}, tz={self.tz})"

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


class TickImbalanceBar(Bar):

    def get_bar(
        self,
        start_date: str = None,
        end_date: str = None,
        initial_size: int = 1000,
        alpha: float = 0.97,
        max_imbalance: float = 100,
    ):
        return get_tick_imbalance_bar(
            [self.symbol],
            start_date,
            end_date,
            self.tz,
            initial_size,
            alpha,
            max_imbalance,
        )


class VolumeImbalanceBar(Bar):

    def get_bar(
        self,
        start_date: str = None,
        end_date: str = None,
        initial_size: int = 1000,
        alpha: float = 0.97,
        max_imbalance: float = 500,
    ):
        return get_volume_imbalance_bar(
            [self.symbol],
            start_date,
            end_date,
            self.tz,
            initial_size,
            alpha,
            max_imbalance,
        )


class DollarImbalanceBar(Bar):

    def get_bar(
        self,
        start_date: str = None,
        end_date: str = None,
        initial_size: int = 1000,
        alpha: float = 0.97,
        max_imbalance: float = 1000000,
    ):
        return get_dollar_imbalance_bar(
            [self.symbol],
            start_date,
            end_date,
            self.tz,
            initial_size,
            alpha,
            max_imbalance,
        )
