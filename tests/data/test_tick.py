import unittest
import pandas as pd

from onetrading.data.tick import get_tick


class TestGetTickDataRaw(unittest.TestCase):

    def test_get_tick_data_raw_symbol_ESH4(self):
        # Call the function
        result_df = get_tick(
            symbol=["ESH4"], start_date="2024-02-01", end_date="2024-02-02"
        )

        # Assert conditions based on the mocked response
        self.assertTrue(not result_df.empty)
        self.assertEqual(
            list(result_df.columns), ["ts_event", "symbol", "price", "size"]
        )
        self.assertEqual(result_df.iloc[0]["symbol"], "ESH4")
        self.assertEqual(
            result_df.iloc[0]["ts_event"],
            pd.Timestamp("2024-02-01 00:00:01.085058+0000", tz="UTC"),
        )
        self.assertEqual(
            result_df.iloc[-1]["ts_event"],
            pd.Timestamp("2024-02-02 21:59:59.941415+0000", tz="UTC"),
        )

    def test_get_tick_data_tz(self):
        # Call the function
        result_df = get_tick(
            symbol=["ESH4"],
            start_date="2024-02-01",
            end_date="2024-02-01",
            tz="US/Eastern",
        )

        # Assert conditions based on the mocked response
        self.assertTrue(not result_df.empty)
        self.assertEqual(
            list(result_df.columns), ["ts_event", "symbol", "price", "size"]
        )
        self.assertEqual(result_df.iloc[0]["symbol"], "ESH4")
        self.assertEqual(
            result_df.iloc[0]["ts_event"],
            pd.Timestamp("2024-02-01 00:00:01.201744-0500", tz="US/Eastern"),
        )
        self.assertEqual(
            result_df.iloc[-1]["ts_event"],
            pd.Timestamp("2024-02-01 23:59:01.794494-0500", tz="US/Eastern"),
        )


if __name__ == "__main__":
    unittest.main()
