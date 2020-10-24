import numpy as np
from numpy import array
import pandas as pd
from pandas import DataFrame
import levy
import statsmodels.tsa.stattools as smt

import os
from datetime import datetime


def partial_autocorrelation(x, n_lags):
    """Gets Partial Autocorrelation from column in DataFrame."""
    corr_lag = smt.pacf(x, nlags=n_lags)
    return corr_lag


def get_q(df: DataFrame) -> array:
    """Gets Q matrix in the form of a (2,2) array."""

    q = np.zeros((2, 2), dtype=float)
    n = len(df.index)
    n_plus = 0
    n_minus = 0

    for i in range(n - 1):
        # plus starting column
        if df.iloc[i, 6] > 0:

            if df.iloc[i + 1, 6] > 0:
                q[0, 0] += 1
            if df.iloc[i + 1, 6] < 0:
                q[1, 0] += 1
            if df.iloc[i + 1, 6] != 0:
                n_plus += 1

        # minus starting column
        if df.iloc[i, 6] < 0:

            if df.iloc[i + 1, 6] > 0:
                q[0, 1] += 1
            if df.iloc[i + 1, 6] < 0:
                q[1, 1] += 1
            if df.iloc[i + 1, 6] != 0:
                n_minus += 1

    q[0, 0] /= n_plus
    q[1, 0] /= n_plus
    q[0, 1] /= n_minus
    q[1, 1] /= n_minus

    return q


def fit_levy_par(x) -> array:
    """Fits Levy-Stable parameters from column in DataFrame."""
    levp = levy.fit_levy(x)
    lp = levp[0].get('0')
    return lp


def get_ticker_name(s: str) -> str:
    """Gets ticker name from file name."""
    x = s.find('_')
    name = s[:x]
    return name


# MAIN FLOW
if __name__ == '__main__':
    now_time = datetime.now()

    # INPUTS
    start = datetime.fromisoformat('2010-06-08')
    end = datetime.fromisoformat('2020-06-08')
    path_data = f'D:/noisy market model/source/analysis/data/{start.date()}___{end.date()}'
    lagdays = 7  # max partial autocorrelation lag

    levy_markers = ['Symbol', 'Alpha', 'Beta', 'Mu', 'Sigma']
    levy_df = DataFrame(columns=levy_markers)

    pacf_markers = ['Symbol']
    for num in range(lagdays + 1):
        pacf_markers.append(f'lag={num}')
    pacf_df = DataFrame(columns=pacf_markers)

    q_markers = ['Symbol', 'Q[0,0]', 'Q[1,0]', 'Q[0,1]', 'Q[1,1]']
    q_df = DataFrame(columns=q_markers)

    # get data from files
    files = os.listdir(path_data)
    for file in files:
        full_name = os.path.join(path_data, file)
        data = pd.read_csv(full_name)

        if str(data.iloc[0, 0]) == str(start.date()):
            ticker = get_ticker_name(file)

            if not data['%Change'].isnull().values.any():

                # Levy Stable Fit
                levy_par = fit_levy_par(data['%Change'])
                levy_list = [ticker]
                for par in levy_par:
                    levy_list.append(par)
                levy_df.loc[len(levy_df)] = levy_list

                # Partial Autocorrelation
                pacf_data = partial_autocorrelation(data['State Sum'], lagdays)
                pacf_list = [ticker]
                for num in pacf_data:
                    pacf_list.append(num)
                pacf_df.loc[len(pacf_df)] = pacf_list

                # Q matrix data
                q_matrix = get_q(data)
                q_matrix = q_matrix.transpose()
                q_matrix = q_matrix.flatten()
                q_list = [ticker]
                for element in q_matrix:
                    q_list.append(element)
                q_df.loc[len(q_df)] = q_list

            else:
                print(f'{ticker}: bad data')

    # write
    levy_df.to_csv(f'Levy-Stable___{start.date()}___{end.date()}.csv', index=False)
    pacf_df.to_csv(f'pacf___{start.date()}___{end.date()}.csv', index=False)
    q_df.to_csv(f'Q-Matrix___{start.date()}___{end.date()}.csv', index=False)

    # timing
    finish_time = datetime.now()
    duration = finish_time - now_time
    minutes, seconds = divmod(duration.seconds, 60)
    print(f'\nThe script took {minutes} minutes and {seconds} seconds to run.')
