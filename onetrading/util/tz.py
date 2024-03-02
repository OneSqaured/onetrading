from datetime import datetime

import pytz


def convert_to_utc(date_str: str, original_tz_str: str):
    # Parse the date string to a datetime object (assuming date_str is in 'YYYY-MM-DD' format)
    my_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Get the original timezone
    original_tz = pytz.timezone(original_tz_str)

    # Localize the datetime object to the original timezone
    my_date_localized = original_tz.localize(my_date)

    # Convert localized datetime to UTC
    my_date_utc = my_date_localized.astimezone(pytz.utc)

    return my_date_utc
