import os
import databento
import pandas as pd
import sqlite3

PATHS = [
    os.path.expanduser("~/Documents/Data/CME_ES_TBBO_202310_202401"),
    os.path.expanduser("~/Documents/Data/CME_ES_TBBO_202402"),
]

# Create an empty list to store the dataframes
result = []

# Use os.listdir to get the list of files in the directory
for PATH in PATHS:
    files = os.listdir(PATH)
    files.sort()

    # Iterate over the files
    for filename in files:
        if not filename.endswith(".zst"):
            continue
        # Join the directory path with the filename
        full_path = os.path.join(PATH, filename)
        # Load the file and convert it to a dataframe
        result.append(databento.DBNStore.from_file(full_path).to_df())

result = pd.concat(result, ignore_index=True)

# store in sqlite
tick = result[["ts_event", "symbol", "price", "size"]]
conn = sqlite3.connect(os.path.expanduser("~/Documents/Data/onesquared.db"))

# Optionally adjust SQLite settings for performance
conn.execute("PRAGMA journal_mode=WAL;")  # Switch to Write-Ahead Logging
conn.execute(
    "PRAGMA cache_size = -10000;"
)  # Set cache size to 10000 pages (default page size is 4KB)

tick.to_sql("tick", conn, if_exists="replace", index=False, chunksize=1000)

conn.close()
