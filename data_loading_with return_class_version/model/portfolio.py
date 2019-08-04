import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Portfolio:

    def __init__(self, df_open, df_high, df_low, df_close, df_adj_close, df_volume, df_return):

        self.df_open = df_open
        self.df_high = df_high
        self.df_low = df_low
        self.df_close = df_close
        self.df_adj_close = df_adj_close
        self.df_volume = df_df_volume
        self.df_return = df_df_return

        self.start_of_week = None
        self.start_of_month = None

    def get_start_of_week(self):
        self.start_of_week = []
        previous = self.df_adj_close.index[0].weekday() - 1
        for t in self.df_adj_close.index:
            now = t.weekday() 
            if now - previous < 0:
                self.start_of_week.append(t)
            previous = now
            
    def get_start_of_month(self):
        self.start_of_month = []
        previous = 1
        for t in df_adj_close.index:
            now = t.month
            if now != previous:
                self.start_of_month.append(t)
            previous = now
            
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







