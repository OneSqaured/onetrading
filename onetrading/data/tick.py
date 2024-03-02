import datetime as dt
from onetrading.data import get_df_from_db
from onetrading.util.tz import convert_to_utc
import pandas as pd


def get_tick(symbol: list[str] = None, start_date: str = None, end_date: str = None, tz: str = "UTC") -> pd.DataFrame:
    """
    Fetches raw tick data from a database based on the specified conditions.

    This function queries a database for tick data, filtering by symbol,
    start date, and end date. It adjusts the input dates to UTC for the query,
    and converts the resulting timestamps to the desired timezone.

    Parameters:
    - symbol (list[str], optional): A list of ticker symbols to filter the data. Defaults to None.
    - start_date (str, optional): The start date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - end_date (str, optional): The end date for the query in 'YYYY-MM-DD' format. Defaults to None.
    - tz (str, optional): The timezone to convert the timestamps to. Defaults to "UTC".

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered tick data, with timestamps
      adjusted to the specified timezone.

    Examples:
    - Fetching data for symbol "ESM4" from 2024-02-01 to 2024-02-02:
        df = get_tick(symbol=["ESM4"], start_date="2024-02-01", end_date="2024-02-02")
        print(df.head())

    - Fetching data for symbol "ESH4" from 2024-02-03 to 2024-02-04, converting timestamps to US/Eastern timezone:
        df = get_tick(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern")
        print(df.tail())

    - Fetching all available tick data without filtering:
        df = get_tick()
        print(df.head())
    """

    # Base query
    query = "SELECT * FROM tick"
    conditions = []

    # Add conditions based on arguments
    if symbol is not None and symbol:  # Checks if symbol is not None and not an empty list
        symbols_str = ", ".join(f"'{s}'" for s in symbol)  # Safe way to format symbols for SQL query
        conditions.append(f"symbol IN ({symbols_str})")

    if start_date is not None:
        start_date = convert_to_utc(start_date, tz).strftime('%Y-%m-%d %H:%M:%S')
        conditions.append(f"ts_event >= '{start_date}'")

    if end_date is not None:
        end_date = convert_to_utc(end_date, tz) + dt.timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
        conditions.append(f"ts_event < '{end_date}'")

    # Append conditions to the query if there are any
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY symbol, ts_event"

    df = get_df_from_db(query)
    if len(df) > 0:
        df["ts_event"] = pd.to_datetime(df["ts_event"], format="ISO8601").dt.tz_convert(tz)

    return df


if __name__ == '__main__':
    df = get_tick(symbol=["ESM4"], start_date="2024-02-01", end_date="2024-02-02")
    print(df.head())
    df = get_tick(symbol=["ESH4"], start_date="2024-02-03", end_date="2024-02-04", tz="US/Eastern")
    print(df.tail())
    df = get_tick()
    print(df.head())
