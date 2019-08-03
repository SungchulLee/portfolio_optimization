import pandas as pd
import os


def data_loading(data_dir='data/dow30_small', benchmark='SPY'):
    
    csv_file_list = os.listdir(data_dir)
    if csv_file_list[0] == '.DS_Store': 
        csv_file_list = csv_file_list[1:]
        
    ticker_list = [] 
    for csv_file in csv_file_list:
        ticker_list.append(csv_file.replace(".csv", ""))
    if benchmark in ticker_list:
        ticker_list.remove(benchmark)
        
    for data_type in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        csv_file_path = os.path.join(data_dir, benchmark + ".csv")
        df = pd.read_csv(csv_file_path,
                         index_col="Date",
                         parse_dates=True,
                         usecols=["Date", data_type],
                         na_values=["null"]).rename(columns={data_type: benchmark})
        if data_type == "Open":
            df_open = df
        if data_type == "High":
            df_high = df   
        if data_type == "Low":
            df_low = df    
        if data_type == "Close":
            df_close = df
        if data_type == "Adj Close":
            df_adj_close = df
        if data_type == "Volume":
            df_volume = df 
    
    for ticker in ticker_list:
        for data_type in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            csv_file_path = os.path.join(data_dir, ticker + ".csv")   
            df_temp = pd.read_csv(csv_file_path,
                                  index_col="Date",
                                  parse_dates=True,
                                  usecols=["Date", data_type],
                                  na_values=["null"]).rename(columns={data_type: ticker})
            if data_type == "Open":
                df_open = df_open.join(df_temp, how='outer')
            if data_type == "High":
                df_high = df_high.join(df_temp, how='outer')   
            if data_type == "Low":
                df_low = df_low.join(df_temp, how='outer')    
            if data_type == "Close":
                df_close = df_close.join(df_temp, how='outer')
            if data_type == "Adj Close":
                df_adj_close = df_adj_close.join(df_temp, how='outer')
            if data_type == "Volume":
                df_volume = df_volume.join(df_temp, how='outer')

    return df_open, df_high, df_low, df_close, df_adj_close, df_volume
