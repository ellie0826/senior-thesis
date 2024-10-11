import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from regime_detection_tf import regime_detection_tf
from regime_prediction import regime_prediction_ml
from exploratory_data_analysis import financial_data
import os
import pickle
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta

class TrendFollowingPortfolio:
    def __init__(self, financial_data, tickers, latest_inception_date, short_window=3, long_window=10, model_output_filename='model_output.pkl'):
        """
        Initialize the TrendFollowingPortfolio with financial data, tickers of interest, a prediction model, and optional leverage.
        """
        self.financial_data = financial_data
        self.commodity_ticker = financial_data.ticker
        self.tickers = tickers
        self.latest_inception_date = latest_inception_date
        if os.path.exists(model_output_filename):
            with open(model_output_filename, 'rb') as file:
                self.prediction_model = pickle.load(file)
        else:
            self.prediction_model = self.load_prediction_model()
            self.prediction_model.run()  # Assuming your model has a .run() method to generate outputs
            with open(model_output_filename, 'wb') as file:
                pickle.dump(self.prediction_model, file)
        self.asset_returns = self.fetch_returns().loc[self.latest_inception_date:]
        self.short_window = short_window
        self.long_window = long_window
        self.weights_dict = {}
        self.w_n = len(tickers)+1
        self.last_rebalance_date = None  
        self.last_weights = np.zeros(self.w_n) 

    def load_prediction_model(self):
        lambda_vals = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
        model_gsci = regime_detection_tf(self.financial_data, lambda_vals)
        model_gsci.run_model(tuning=True)
        # model_gsci.split_data(train_size=0.8)

        macro_data_path = "current.csv"
        model_type = 'logistic_regression'
        model = regime_prediction_ml(model_gsci, macro_data_path, model_type=model_type, use_macro_data=True)

        return model

    def fetch_returns(self):
        """
        Fetch and return the historical adjusted close prices and calculate returns for the assets.
        """
        other_data = yf.download(self.tickers, start=self.financial_data.start_date, end=self.financial_data.end_date, interval=self.financial_data.interval)['Adj Close']
        other_returns = other_data.pct_change().dropna()

        primary_returns = self.financial_data.preprocess_data()
        primary_returns.index = pd.to_datetime(primary_returns.index, format="%Y-%m-%d").round('D')
        primary_returns = primary_returns['Daily Return']

        combined_returns = pd.concat([primary_returns, other_returns], axis=1).dropna()
        return combined_returns

    def determine_regime(self, current_date):
        """
        Determine the market regime for a given date based on the prediction model.
        """
        if current_date in self.prediction_model.final_df.index:
            regime = self.prediction_model.final_df.loc[current_date, 'predicted regime label']
            crash_prob = self.prediction_model.final_df.loc[current_date, 'final probabilities']
            growth_prob = 1 - crash_prob
        else:
            regime = None
            crash_prob, growth_prob = 0, 0
        return regime, crash_prob, growth_prob
    
    def should_rebalance(self, current_date):
        """
        Determine if the portfolio should be rebalanced based on the 6-month frequency.
        """
        if self.last_rebalance_date is None:
            return True
        return current_date >= self.last_rebalance_date + timedelta(days=30)  

    def generate_signals(self, current_date, regime_switching = True):
        
        current_date_dt = pd.to_datetime(current_date)
        signals = np.zeros(self.w_n)  

        short_ma = self.asset_returns.rolling(window=self.short_window).mean().shift(1).dropna()
        long_ma = self.asset_returns.rolling(window=self.long_window).mean().shift(1).dropna()
        
        if self.should_rebalance(current_date_dt):
            if regime_switching:
                regime, crash_prob, growth_prob = self.determine_regime(current_date)
                commodity_signal = 1 + growth_prob if regime == 'E' else 1 - crash_prob
            else:
                commodity_signal = 1

            if current_date in short_ma.index and current_date in long_ma.index:
                for i, ticker in enumerate([self.commodity_ticker] + self.tickers):
                    short_avg = short_ma.loc[current_date_dt].iloc[i]
                    long_avg = long_ma.loc[current_date_dt].iloc[i]
                    
                    if short_avg > long_avg:
                        signals[i] = 1 * (commodity_signal if i == 0 else 1)
                    elif short_avg < long_avg:
                        decrease_factor = 0.6
                        signals[i] = decrease_factor * (commodity_signal if i == 0 else 1)
                        # signals[i] = -1 * (commodity_signal if i == 0 else 1)
                    else:
                        signals[i] = 0
            
            # Normalize signals to weights only if signals were updated
            abs_sum_signals = np.sum(np.abs(signals))
            if abs_sum_signals > 0:
                self.last_weights = signals / abs_sum_signals
            self.last_rebalance_date = current_date_dt
        else:
            # Use last_weights if not rebalancing
            signals = self.last_weights
        
        self.weights_dict[current_date_dt] = self.last_weights
        return self.last_weights

    def backtest(self, regime_switching=True):
        """
        Modified to first store weights for each date, then calculate returns based on these weights.
        """
        strategy_returns = []
        daily_returns = self.asset_returns.loc[self.latest_inception_date:]

        for date in daily_returns.index:
            signals = self.generate_signals(date, regime_switching=regime_switching)
            daily_return = np.dot(daily_returns.loc[date], signals)
            strategy_returns.append(daily_return)

        strategy_returns = np.array(strategy_returns)
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1

        return cumulative_returns, strategy_returns


    def calculate_sharpe_ratio(self, returns, risk_free_rate=0):
        """
        Calculate the Sharpe ratio of a strategy.
        :param returns: numpy array of daily returns.
        :param risk_free_rate: daily risk-free rate, default is 0.
        :return: Sharpe ratio.
        """
        excess_returns = returns - risk_free_rate
        annualized_excess_return = np.mean(excess_returns) * 12
        print('excess return', annualized_excess_return)
        annualized_volatility = np.std(excess_returns) * np.sqrt(12)
        print('annualized vol', annualized_volatility)
        sharpe_ratio = annualized_excess_return / annualized_volatility
        return sharpe_ratio

    def calculate_calmar_ratio(self, returns):
        """
        Calculate the Calmar ratio for a strategy.
        :param returns: numpy array of daily returns.
        :return: Calmar ratio.
        """
        # Calculate cumulative wealth from returns
        cumulative_returns = np.cumprod(1 + returns, axis=0)
        
        # Compute the running maximum
        cumulative_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = 1 - (cumulative_returns / cumulative_max)
        
        index_max_drawdown = np.argmax(drawdowns)
        index_peak = np.argmax(cumulative_returns[:index_max_drawdown+1])
        
        # Compute the maximum drawdown 
        mdd = (1 - cumulative_returns[index_max_drawdown] / cumulative_returns[index_peak]) 
        print('maximum drawdown', mdd)

        annualized_return = np.mean(returns) * 12

        calmar_ratio = annualized_return / abs(mdd) if mdd != 0 else np.nan
        
        return calmar_ratio

    def calculate_sortino_ratio(self, returns, target_return=0):
        """
        Calculate the Sortino ratio of a strategy.
        :param returns: numpy array of daily returns.
        :param target_return: the target or required rate of return, default is 0.
        :return: Sortino ratio.
        """
        downside_returns = returns[returns < target_return]
        annualized_return = np.mean(returns) * 12
        annualized_downside_std = np.std(downside_returns) * np.sqrt(12)
        sortino_ratio = (annualized_return - target_return) / annualized_downside_std
        return sortino_ratio
    
    # def compare_with_sp500(self):
    #     # Fetch S&P 500 monthly data
    #     sp500_data = yf.download('SPY', start=self.financial_data.start_date, end=self.financial_data.end_date, interval='1mo')['Adj Close']
    #     sp500_returns = sp500_data.pct_change().dropna()

    #     # Calculate metrics
    #     cumulative_returns = np.cumprod(1 + sp500_returns) - 1
    #     sharpe_ratio = self.calculate_sharpe_ratio(sp500_returns.values)
    #     sortino_ratio = self.calculate_sortino_ratio(sp500_returns.values)
    #     calmar_ratio = self.calculate_calmar_ratio(cumulative_returns.values)
    #     max_drawdown = self.calculate_maximum_drawdown(cumulative_returns.values)
    #     mean_return = sp500_returns.mean()
    #     std_deviation = sp500_returns.std()

    #     # Display metrics
    #     print("S&P 500 Performance Metrics (Monthly):")
    #     print(f"Mean Return: {mean_return}")
    #     print(f"Standard Deviation: {std_deviation}")
    #     print(f"Sharpe Ratio: {sharpe_ratio}")
    #     print(f"Sortino Ratio: {sortino_ratio}")
    #     print(f"Calmar Ratio: {calmar_ratio}")
    #     print(f"Maximum Drawdown: {max_drawdown}")

    #     # Optionally, plot S&P 500 cumulative returns for visual comparison
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(cumulative_returns, label='S&P 500')
    #     plt.title("Cumulative Returns for S&P 500 (Monthly)")
    #     plt.xlabel("Time")
    #     plt.ylabel("Cumulative Returns")
    #     plt.legend()
    #     plt.show()

    #     return mean_return, std_deviation, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown


    def backtest_equal_weights(self):
        """
        Backtest using equal weights for each asset in the portfolio.
        """
        daily_returns = self.asset_returns
        equal_weights = np.ones(self.w_n) / self.w_n
        strategy_returns = []

        for date in daily_returns.index:
            daily_return = np.dot(daily_returns.loc[date], equal_weights)
            strategy_returns.append(daily_return)

        strategy_returns = np.array(strategy_returns)
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1

        return strategy_returns, cumulative_returns
    

    def run(self):
        """
        Execute the backtesting and calculate performance metrics.
        """
        cumulative_returns, strategy_returns = self.backtest()
        sharpe_ratio = self.calculate_sharpe_ratio(strategy_returns)
        calmar_ratio = self.calculate_calmar_ratio(strategy_returns)
        sortino_ratio = self.calculate_sortino_ratio(strategy_returns)
        print(f"Cumulative Returns: {cumulative_returns[-1]:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")

        cumulative_returns_nr, strategy_returns_nr = self.backtest(regime_switching=False)
        sharpe_ratio_nr = self.calculate_sharpe_ratio(strategy_returns_nr)
        calmar_ratio_nr = self.calculate_calmar_ratio(strategy_returns_nr)
        sortino_ratio_nr = self.calculate_sortino_ratio(strategy_returns_nr)
        print(f"Cumulative Returns Without Regime Switching: {cumulative_returns_nr[-1]:.2f}")
        print(f"Sharpe Ratio Without Regime Switching: {sharpe_ratio_nr:.2f}")
        print(f"Calmar Ratio Without Regime Switching: {calmar_ratio_nr:.2f}")
        print(f"Sortino Ratio Without Regime Switching: {sortino_ratio_nr:.2f}")

        benchmark_returns = 0.6 * self.asset_returns[tickers[0]] + 0.4 * self.asset_returns[tickers[1]]
        benchmark_cumulative_returns = np.cumprod(1 + benchmark_returns) - 1

        benchmark_sharpe = self.calculate_sharpe_ratio(benchmark_returns)
        benchmark_calmar = self.calculate_calmar_ratio(benchmark_returns)
        benchmark_sortino = self.calculate_sortino_ratio(benchmark_returns)

        print(f"\n60/40 Benchmark Metrics:")
        print(f"Sharpe Ratio: {benchmark_sharpe:.2f}")
        print(f"Calmar Ratio: {benchmark_calmar:.2f}")
        print(f"Sortino Ratio: {benchmark_sortino:.2f}")

        # Equal Weights Portfolio
        equal_weight_returns, equal_weight_cumulative_returns = self.backtest_equal_weights()
        equal_weight_sharpe = self.calculate_sharpe_ratio(equal_weight_returns)
        equal_weight_calmar = self.calculate_calmar_ratio(equal_weight_returns)
        equal_weight_sortino = self.calculate_sortino_ratio(equal_weight_returns)
        print(f"\nEqual Weights Portfolio Metrics:")
        print(f"Sharpe Ratio: {equal_weight_sharpe:.2f}")
        print(f"Calmar Ratio: {equal_weight_calmar:.2f}")
        print(f"Sortino Ratio: {equal_weight_sortino:.2f}")

        return cumulative_returns, sharpe_ratio


def find_latest_inception_date(tickers, commodity_tickers, start_date, interval):
    all_tickers = set(tickers for portfolio in portfolios for tickers in portfolio) | set(commodity_tickers)
    latest_inception_date = pd.to_datetime(start_date)

    for ticker in all_tickers:
        data = yf.download(ticker, start=start_date, interval=interval)
        first_valid_index = data.first_valid_index()
        if first_valid_index is not None and first_valid_index > latest_inception_date:
            latest_inception_date = first_valid_index

    return latest_inception_date


# Define commodity tickers and portfolios
commodity_tickers = ['^SPGSCI', 'DBC']
portfolios = [
    ['SPY', 'TLT'],
    ['SPY', 'TLT', '^SBBMGLU'],
    ['SPY', 'TLT', 'IEF', 'SHY', 'XSVM'],
    ['SPY', 'TLT', 'IEF', 'SHY', 'XSVM', '^SBBMGLU', 'IGSB']
]

interval = '1mo'
global_start_date = "1990-01-01" 
latest_inception_date = find_latest_inception_date(portfolios, commodity_tickers, global_start_date, interval)
print('latest outside class', latest_inception_date)


# Loop through each commodity ticker
for commodity_ticker in commodity_tickers:
    # Adjust start_date based on the latest_inception_date
    start_date = "1991-05-01" if commodity_ticker == '^SPGSCI' else "2006-02-03"
    commodity_data = financial_data(commodity_ticker, start_date, interval)
    
    # Loop through each portfolio
    for tickers in portfolios:
        print(f"Running for Commodity Ticker: {commodity_ticker} with Portfolio: {tickers}")
        
        # Adjust start date
        trend_following_portfolio = TrendFollowingPortfolio(commodity_data, tickers, latest_inception_date)
        # Run the analysis
        trend_following_portfolio.run()
        
        print("\nFinished running for the current configuration.\n")


# interval = "1mo"

# ticker = "DBC"
# start_date = "2006-02-03"

# commodity = financial_data(ticker, start_date, interval)
# # lambda_vals = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
# # model_gsci = regime_detection_tf(gsci, lambda_vals)
# # model_gsci.run_model(tuning=True)

# # macro_data_path = "current.csv"
# # model_type = 'logistic_regression'
# # gsci_prediction = regime_prediction_ml(model_gsci, macro_data_path, model_type=model_type, use_macro_data=True)
# # gsci_prediction.run()

# tickers = ['SPY', 'TLT', 'IEF', 'SHY', 'XSVM', '^SBBMGLU', 'IGSB'] 
# trend_following_portfolio = TrendFollowingPortfolio(commodity, tickers)
# trend_following_portfolio.run()
