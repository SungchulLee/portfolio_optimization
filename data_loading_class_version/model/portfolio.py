import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Portfolio:

    def __init__(self, data_dir, benchmark='SPY', start="2017-01-01", end="2017-12-31"):

        self.data_dir = data_dir
        self.benchmark = benchmark
        self.start = start
        self.end = end

        self.ticker_list = None
        self.open, self.high, self.low, self.close, self.adj_close, self.volume = None, None, None, None, None, None
        self.normalized_adj_close = None
        self.daily_return = None
        self.sigma, self.mu = None, None

        self.data_loading()

    def data_loading(self):

        dates = pd.date_range(self.start, self.end)

        csv_file_list = os.listdir(self.data_dir)
        if csv_file_list[0] == '.DS_Store':
            csv_file_list = os.listdir(self.data_dir)[1:]

        self.ticker_list = []
        for csv_file in csv_file_list:
            self.ticker_list.append(csv_file.replace(".csv", ""))

        for data_type in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:

            df = pd.DataFrame(index=dates)
            for ticker in self.ticker_list:
                csv_file_path = os.path.join(self.data_dir, ticker + ".csv")

                df_temp = pd.read_csv(csv_file_path,
                                      index_col="Date",
                                      parse_dates=True,
                                      usecols=["Date", data_type],
                                      na_values=["null"])

                df_temp = df_temp.rename(columns={data_type: ticker})
                df = df.join(df_temp)

                if ticker == self.benchmark:
                    df = df.dropna(subset=[self.benchmark])

            if data_type == "Open":
                self.open = df
            if data_type == "High":
                self.high = df
            if data_type == "Low":
                self.low = df
            if data_type == "Close":
                self.close = df
            if data_type == "Adj Close":
                self.adj_close = df
            if data_type == "Volume":
                self.volume = df

    def compute_normalized_adj_close(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        data = self.adj_close[start:end]
        self.normalized_adj_close = data / data.iloc[0, :]

    def compute_daily_return(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        data = self.adj_close[start:end]
        self.daily_return = data.pct_change()

    def compute_sigma_and_mu(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        self.compute_daily_return(start=start, end=end)
        self.sigma = 252 * self.daily_return.cov()
        self.mu = 252 * self.daily_return.mean()

    def plot_risk_return(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        self.compute_sigma_and_mu(start=start, end=end)

        fig, ax = plt.subplots()

        for ticker in self.ticker_list:
            ax.scatter(np.sqrt(self.sigma.loc[ticker, ticker]), self.mu[ticker], alpha=0.5)
            ax.annotate(ticker, (np.sqrt(self.sigma.loc[ticker, ticker]), self.mu[ticker]))

        plt.show()

    def equal_portfolio(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        self.compute_normalized_adj_close(start=start, end=end)

        n = len(self.ticker_list)
        w = np.ones((n,)) / n
        adj_close = np.sum(w * self.normalized_adj_close, axis=1)
        daily_return = adj_close.pct_change()
        sigma = 252 * daily_return.var()
        mu = 252 * daily_return.mean()

        return adj_close, daily_return, sigma, mu

    def gmvp(self, start=None, end=None):

        if start is None:
            start = self.start
        if end is None:
            end = self.end

        if self.adj_close is None:
            raise ValueError("Please run data_loading method first")

        self.compute_normalized_adj_close(start=start, end=end)

        n = len(self.ticker_list)
        w = np.ones((n,)) / n
        adj_close = np.sum(w * self.normalized_adj_close, axis=1)
        daily_return = adj_close.pct_change()
        sigma = 252 * daily_return.var()
        mu = 252 * daily_return.mean()

        return adj_close, daily_return, sigma, mu







