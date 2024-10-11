import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class financial_data:
    def __init__(self, ticker, start_date, interval, end_date = None) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.df = yf.download(ticker, start=start_date, end = end_date, interval=interval)
        self.price_series = self.df['Adj Close']
    
    def preprocess_data(self):
        df = self.df
        df.dropna(subset=['Adj Close'], inplace=True)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        # Calculate daily returns and log returns
        df['Daily Return'] = df['Adj Close'].pct_change().dropna()
        df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        df['Cumulative Log Return'] = np.cumsum(df['Log Return'])
        df.index = pd.to_datetime(df.index).round('D')

        df = df.dropna()

        return df
    
    def plot_data(self, label):
        df = self.preprocess_data()
        plt.figure(figsize=(12, 6))
        df['Cumulative Log Return'].plot(title=label+ ' Cumulative Log Return Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Daily Log Return')
        plt.show()

        plt.figure(figsize=(16, 8))
        df[['Close', 'Open', 'High', 'Low']].plot(ax=plt.gca())
        plt.title(label + ' Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

        return None

    def compute_max_drawdown(self):
        price_series = self.price_series
        if not isinstance(price_series, pd.Series):
            price_series = pd.Series(price_series)

        window = 12

        # Compute maximum drawdown
        rolling_max = price_series.rolling(window, min_periods=None).max()
        drawdown = price_series/rolling_max - 1.0

        max_drawdown = drawdown.rolling(window, min_periods=None).min()

        # Plot the results
        plt.plot(drawdown, label="Drawdown")
        plt.plot(max_drawdown, label="Maximum Drawdown in a Year")
        plt.show()

        return max_drawdown


# ticker = "^SPGSCI"
# start_date = "1991-05-01"
# interval = "1mo"
# label = "GSCI"

# spgsci = financial_data(ticker, start_date, interval)
# df = spgsci.preprocess_data()

# spgsci.compute_max_drawdown()

