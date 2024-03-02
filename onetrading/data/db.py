import sqlite3
import pandas as pd
import os

DB_PATH = os.path.expanduser('~/Documents/Data/onesquared.db')


def get_df_from_db(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == '__main__':
    query = 'SELECT * FROM tick LIMIT 100'
    df = get_df_from_db(query)
    print(df.head())
