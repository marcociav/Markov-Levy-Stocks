from datetime import datetime
import yfinance as yf

import os
import pandas as pd
from pandas import DataFrame
import numpy as np


def get_stock(stock: str, beg: datetime, fin: datetime):
    """gets stock data and writes .csv file"""
    new_beg = datetime(beg.year, beg.month, beg.day - 1)
    df = yf.download(stock, start=new_beg, end=fin)
    return df


def modify_stock_data(df: DataFrame):
    """ Cleans data, adds Change and %Change columns.
        Returns DataFrame."""
    df = df.drop('Adj Close', 1)

    change = np.zeros(len(df.index))
    for i in range(1, len(df.index)):
        change[i] = df.iloc[i, 3] - df.iloc[i - 1, 3]
    df.insert(4, 'Change', change)

    change_perc = np.zeros(len(df.index))
    state_cum = np.zeros(len(df.index))
    for i in range(1, len(df.index)):
        change_perc[i] = df.iloc[i, 4] / df.iloc[i - 1, 3]
        if i == 1:
            state_cum[i] = np.sign(change_perc[i])
        if i != 1:
            state_cum[i] = state_cum[i-1] + np.sign(change_perc[i])
    df.insert(5, '%Change', change_perc)
    df.insert(6, 'State Sum', state_cum)
    df = df[1:]

    return df


# MAIN FLOW
if __name__ == '__main__':

    now_time = datetime.now()
    # bad_stocks = []

    # INPUTS
    snp = pd.read_csv('S&P500-Symbols.csv')
    start = datetime(2010, 6, 8)
    end = datetime(2020, 6, 8)

    # write files
    out_dir = f'./{start.date()}___{end.date()}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for s in snp['Symbol']:
        # try:
            print(f"${s}")
            data = get_stock(s, start, end)
            data = modify_stock_data(data)

            out_name = f'{s}___{start.date()}___{end.date()}.csv'
            full_name = os.path.join(out_dir, out_name)
            data.to_csv(full_name)
        # except:
        #     print(f'Bad name or not enough data for {s}')
        #     bad_stocks.append(s)

    # print(f'Bad stocks: {len(bad_stocks)}')
    # for k in range(len(bad_stocks)):
    #     print(bad_stocks[k])

    # timing:
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print(f'\nThe script took {minutes} minutes and {seconds} seconds to run.')
