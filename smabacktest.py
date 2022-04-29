# %%
import pandas as pd
import numpy as np
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt

class SMABacktester():
    
    def __init__(self, symbol, SMA_S, SMA_L, start_date, end_date):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start_date = start_date
        self.end_date = end_date
        self.results = None
        self.get_data()
    
    def get_data(self):
        df = yf.download(self.symbol, start_date=self.start_date, end_date=self.end_date)
        # Get the close price and convert to dataframe
        data = df.Close.to_frame()
        data["returns"] = np.log(data.Close.div(data.Close.shift(1)))
        data["SMA_S"] = data.Close.rolling(self.SMA_S).mean()
        data["SMA_L"] = data.Close.rolling(self.SMA_L).mean()
        data.dropna(inplace=True)
        self.data2 = data
        
        return data
    
    def test_results(self):
        data = self.data2.copy().dropna()
        data["position"] = np.where(data["SMA_S"]>data["SMA_L"],1,-1)
        data["strategy"] = data["returns"]*data.position.shift(1)
        data.dropna(inplace=True)
        # Buy and Hold Returns
        data["returns_bh"] = data["returns"].cumsum().apply(np.exp)
        data["returns_strategy"] = data["strategy"].cumsum().apply(np.exp)
        # Last Row
        performance = data["returns_strategy"].iloc[-1]
        # Outperformance
        outperformance = performance - data["returns_bh"].iloc[-1]
        self.results = data

        ret = np.exp(data["strategy"].sum())
        std = data["strategy"].std()*np.sqrt(252)
        
        #return ret, std
        return round(performance, 6), round(outperformance, 6)
    
    def plot_results(self):
        if self.results is None:
            print("Run the test, please")
        else:
            title = "{}| SMA_S={} | SMA_L{}".format(self.symbol, self.SMA_S, self.SMA_L)
            self.results[["returns_bh", "returns_strategy"]].plot(title=title, figsize=(12,8))
        

# %%



