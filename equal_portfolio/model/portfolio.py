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
        self.df_volume = df_volume
        self.df_return = df_return

        self.start_of_week = None
        self.start_of_month = None
        self.my_account_initial = None
        self.my_account_final = None

    def get_start_of_week(self):
        self.start_of_week = [self.df_adj_close.index[0]]
        previous = self.df_adj_close.index[0].weekday() 
        for t in self.df_adj_close.index[1:]:
            now = t.weekday() 
            if now - previous < 0:
                self.start_of_week.append(t)
            previous = now
            
    def get_start_of_month(self):
        self.start_of_month = [self.df_adj_close.index[0]]
        previous = self.df_adj_close.index[0].month
        for t in self.df_adj_close.index:
            now = t.month
            if now != previous:
                self.start_of_month.append(t)
            previous = now
            
    def get_estimation_period_data(self):
        self.get_start_of_week()
        n = len(self.start_of_week)
        start_estimation = self.start_of_week[1:n-5]
        end_estimation = self.start_of_week[5:n-1] 
        start_deployment = self.start_of_week[5:n-1]
        end_deployment = self.start_of_week[6:n]
        for start_e, end_e, start_d, end_d in zip(start_estimation, end_estimation, start_deployment, end_deployment):
            yield self.df_open[start_e:end_e].iloc[:-1,:], \
            self.df_high[start_e:end_e].iloc[:-1,:], \
            self.df_low[start_e:end_e].iloc[:-1,:], \
            self.df_close[start_e:end_e].iloc[:-1,:], \
            self.df_adj_close[start_e:end_e].iloc[:-1,:], \
            self.df_volume[start_e:end_e].iloc[:-1,:], \
            self.df_return[start_e:end_e].iloc[:-1,:]
            
    def get_deployment_period_data(self):
        self.get_start_of_week()
        n = len(self.start_of_week)
        start_estimation = self.start_of_week[1:n-5]
        end_estimation = self.start_of_week[5:n-1] 
        start_deployment = self.start_of_week[5:n-1]
        end_deployment = self.start_of_week[6:n]
        for start_e, end_e, start_d, end_d in zip(start_estimation, end_estimation, start_deployment, end_deployment):
            yield self.df_open[start_d:end_d].iloc[:-1,:], \
            self.df_high[start_d:end_d].iloc[:-1,:], \
            self.df_low[start_d:end_d].iloc[:-1,:], \
            self.df_close[start_d:end_d].iloc[:-1,:], \
            self.df_adj_close[start_d:end_d].iloc[:-1,:], \
            self.df_volume[start_d:end_d].iloc[:-1,:], \
            self.df_return[start_d:end_d].iloc[:-1,:]
            
    def make_my_account_initial(self):
        self.my_account_initial = np.zeros(shape=(self.df_adj_close.shape[0], self.df_adj_close.shape[1]))
        self.my_account_initial = pd.DataFrame(self.my_account_initial, index=self.df_adj_close.index, columns=self.df_adj_close.columns)
        self.my_account_initial['CASH'] = 1.0
        self.my_account_initial['TOTAL'] = 1.0
        
    def make_my_account_final(self):
        self.my_account_final = np.zeros(shape=(self.df_adj_close.shape[0], self.df_adj_close.shape[1]))
        self.my_account_final = pd.DataFrame(self.my_account_final, index=self.df_adj_close.index, columns=self.df_adj_close.columns)
        self.my_account_final['CASH'] = 1.0
        self.my_account_final['TOTAL'] = 1.0
        
    def make_my_account_equal(self):
        self.make_my_account_initial()
        self.make_my_account_final()
        for idx, t in enumerate(self.df_adj_close.index):
            weight = np.nan_to_num(self.df_adj_close.iloc[idx,:]) 
            weight[weight!=0] = 1.0 / np.sum(weight!=0)
            if t == self.df_adj_close.index[0]:
                initial_capital = 1.0
            else:
                initial_capital = self.my_account_final['TOTAL'][idx-1]
            self.my_account_initial.iloc[idx, :-2] = weight * initial_capital
            self.my_account_initial.iloc[idx, -2] = 0
            self.my_account_initial.iloc[idx, -1] = initial_capital
            self.my_account_final.iloc[idx, :-2] = self.my_account_initial.iloc[idx, :-2] * (1 + self.df_return.iloc[idx, :])
            self.my_account_final.iloc[idx, -2] = 0
            self.my_account_final.iloc[idx, -1] = np.sum(self.my_account_final.iloc[idx, :-2])
            
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







