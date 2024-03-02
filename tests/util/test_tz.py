import datetime as dt
import unittest
import pytz

from onetrading.util.tz import convert_to_utc


class TestConvertToUtc(unittest.TestCase):

    def test_conversion_positive_offset(self):
        """Test converting from a timezone with a positive UTC offset to UTC."""
        # Example: Converting '2023-03-01' from 'Asia/Kolkata' (UTC+5:30) to UTC
        date_str = "2023-03-01"
        original_tz_str = "Asia/Kolkata"
        expected_utc = dt.datetime(2023, 2, 28, 18, 30, tzinfo=dt.timezone.utc)
        result = convert_to_utc(date_str, original_tz_str)
        self.assertEqual(result, expected_utc)

    def test_conversion_negative_offset(self):
        """Test converting from a timezone with a negative UTC offset to UTC."""
        # Example: Converting '2023-03-01' from 'America/New_York' (UTC-5) to UTC during standard time
        date_str = "2023-03-01"
        original_tz_str = "America/New_York"
        expected_utc = dt.datetime(2023, 3, 1, 5, 0, tzinfo=dt.timezone.utc)
        result = convert_to_utc(date_str, original_tz_str)
        self.assertEqual(result, expected_utc)

    def test_invalid_date_format(self):
        """Test the function with an invalid date format."""
        date_str = "01-03-2023"  # Incorrect format
        original_tz_str = "Europe/London"
        with self.assertRaises(ValueError):
            convert_to_utc(date_str, original_tz_str)

    def test_invalid_timezone(self):
        """Test the function with an invalid timezone string."""
        date_str = "2023-03-01"
        original_tz_str = "Invalid/Timezone"
        with self.assertRaises(pytz.UnknownTimeZoneError):
            convert_to_utc(date_str, original_tz_str)


if __name__ == "__main__":
    unittest.main()
