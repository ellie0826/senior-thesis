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


class RiskParityPortfolio:
    def __init__(self, financial_data, tickers, latest_inception_date, model_output_filename='model_output.pkl'):
        """
        Initialize the RiskParityPortfolio with financial data and tickers of interest.
        """
        self.financial_data = financial_data
        self.commodity_ticker = financial_data.ticker
        self.latest_inception_date = latest_inception_date
        self.tickers = tickers
        if os.path.exists(model_output_filename):
            with open(model_output_filename, 'rb') as file:
                self.prediction_model = pickle.load(file)
        else:
            self.prediction_model = self.load_prediction_model()
            self.prediction_model.run()  # Assuming your model has a .run() method to generate outputs
            with open(model_output_filename, 'wb') as file:
                pickle.dump(self.prediction_model, file)
        self.asset_returns = self.fetch_returns().loc[self.latest_inception_date:]
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
        # Identify the latest start date among all assets
        self.latest_start_date = pd.to_datetime(combined_returns.index.min())

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
            self.last_rebalance_date = current_date
            return True
            
        return current_date >= self.last_rebalance_date + timedelta(days=365)  

    def calculate_covariance_matrix(self, current_date):
        """
        Calculate and return the annualized covariance matrix of the asset returns.
        """
        current_date_dt = pd.to_datetime(current_date)
        daily_returns = self.asset_returns
                # Filter returns for dates corresponding to the given regime up to the current date
        dates = self.prediction_model.final_df[self.prediction_model.final_df.index < current_date_dt].index

        dates = dates[dates > self.latest_inception_date]
        dates_dt = pd.to_datetime(dates).date
        filtered_returns = self.asset_returns.loc[dates_dt]

        if filtered_returns.empty:
            return self.calculate_overall_covariance_matrix()  
        else:
            return filtered_returns.cov() * 12  
    
    def calculate_overall_covariance_matrix(self):
        """
        Calculate and return the annualized covariance matrix of the asset returns.
        """
        daily_returns = self.asset_returns
        covariance_matrix = daily_returns.cov() * 12  
        return covariance_matrix
    
    def calculate_covariance_for_regime(self, regime, current_date):
        # Ensure current_date is a datetime object
        current_date_dt = pd.to_datetime(current_date)

        # Filter returns for dates corresponding to the given regime up to the current date
        regime_dates = self.prediction_model.final_df[
            (self.prediction_model.final_df['predicted regime label'] == regime) & 
            (self.prediction_model.final_df.index < current_date_dt)
        ].index

        regime_dates = regime_dates[regime_dates > self.latest_inception_date]

        # Convert regime_dates to the correct format and ensure it's a datetime index
        regime_dates_dt = pd.to_datetime(regime_dates).date
        # Filter returns based on the valid dates
        filtered_returns = self.asset_returns.loc[regime_dates_dt]

        # Check if filtered_returns is empty
        if filtered_returns.empty:
            print("No data for regime {}, using overall covariance matrix.".format(regime))
            return self.calculate_overall_covariance_matrix()  
        else:
            return filtered_returns.cov() * 12  
    
    def calculate_weighted_covariance_matrix(self, current_date):
        current_date_dt = pd.to_datetime(current_date).strftime('%Y-%m-%d')

        regime, crash_prob, growth_prob = self.determine_regime(current_date_dt)
        cov_matrix_normal = self.calculate_covariance_for_regime('E', current_date_dt)
        cov_matrix_crash = self.calculate_covariance_for_regime('C', current_date_dt)
        weighted_cov_matrix = growth_prob * cov_matrix_normal + crash_prob * cov_matrix_crash
            
        return weighted_cov_matrix

    def optimize_weights(self, covariance_matrix):
        """
        Optimize portfolio weights using the risk parity approach.
        """
        # The number of assets in the portfolio
        num_assets = self.w_n

        # Initial guess for the weights (equal weights)
        initial_weights = np.array(num_assets * [1. / num_assets,])

        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.})

        # Bounds: Weights are bounded between 0 and 1 for each asset to avoid shorting
        bounds = tuple((0, 1) for asset in range(num_assets))

        # Objective function: minimize the portfolio risk subject to the constraints
        def objective(weights): 
            # Calculate portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            # The objective is to minimize variance
            return portfolio_variance

        # Optimization
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise BaseException('Risk Parity optimization did not converge.')
        

        return result.x
            
    def generate_signals(self, current_date, regime_switching=True):
        """
        Generate trading signals, adjusting based on regime probability every 6 months,
        and using risk parity for portfolio optimization.
        """
        current_date_dt = pd.to_datetime(current_date)

        # Check if it's time to rebalance
        if self.should_rebalance(current_date_dt):

            # Select the covariance matrix based on the regime switching flag
            if regime_switching:
                # Make sure there is data for the regime-based covariance matrix
                if current_date_dt >= self.latest_inception_date:
                    covariance_matrix = self.calculate_weighted_covariance_matrix(current_date)
                else:
                    print("Not enough data for regime-based covariance matrix. Falling back to overall covariance matrix.")
                    covariance_matrix = self.calculate_overall_covariance_matrix()
            else:
                if current_date_dt >= self.latest_inception_date:
                    covariance_matrix = self.calculate_covariance_matrix(current_date)
                else:
                    print("Not enough data for regime-based covariance matrix. Falling back to overall covariance matrix.")
                    covariance_matrix = self.calculate_overall_covariance_matrix()
            
            # Optimize weights
            optimized_weights = self.optimize_weights(covariance_matrix)

            # Update last weights and rebalance date
            self.last_weights = optimized_weights
            self.last_rebalance_date = current_date_dt

        else:
            optimized_weights = self.last_weights
        
        self.weights_dict[current_date_dt] = optimized_weights
        return optimized_weights

    def reset_portfolio(self):
        """
        Resets the portfolio's state to initial conditions. This includes resetting the last rebalance date,
        last weights, and any other stateful properties of the portfolio.
        """
        self.last_rebalance_date = None
        self.last_weights = np.zeros(self.w_n)  # Reset to zeros based on the number of tickers + 1 for the commodity ticker
        # You might also want to reset any other stateful attributes here

    # Existing methods...


    def backtest(self, regime_switching=True):
        """
        Perform backtesting, ensuring it starts after the latest start date of asset returns.
        """
        # Ensure test_start_date is a datetime object for comparison
        test_start_date = pd.to_datetime(self.prediction_model.test_start_date)
        backtest_start_date = max(test_start_date, self.latest_start_date)  # Choose the later start date

        strategy_returns = []
        # Filter daily returns to start at backtest_start_date
        # daily_returns_filtered = self.asset_returns[self.asset_returns.index >= backtest_start_date]
        print('backtest start date', backtest_start_date)
        daily_returns_filtered = self.asset_returns.loc[latest_inception_date:]

        for date in daily_returns_filtered.index:
            signals = self.generate_signals(date.strftime('%Y-%m-%d'), regime_switching=regime_switching)
            daily_return = np.dot(daily_returns_filtered.loc[date], signals)
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

    # def calculate_calmar_ratio(self, cumulative_returns):
    #     """
    #     Calculate the Calmar ratio for a strategy.
    #     :param cumulative_returns: numpy array of cumulative returns.
    #     :return: Calmar ratio.
    #     """
    #     # Calculate the annualized return
    #     annualized_return = (cumulative_returns[-1] / cumulative_returns[0]) ** (12 / len(cumulative_returns)) - 1
        
    #     # Calculate the drawdowns
    #     cumulative_max = np.maximum.accumulate(cumulative_returns)
    #     drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
        
    #     # Find the maximum drawdown
    #     max_drawdown = np.min(drawdowns)  # This correctly identifies the max drawdown
        
    #     print('Max Drawdown:', max_drawdown)
        
    #     # Calculate the Calmar ratio
    #     calmar_ratio = annualized_return / abs(max_drawdown)
    #     return calmar_ratio

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
        
        # Identify the index of the maximum drawdown
        index_max_drawdown = np.argmax(drawdowns)
        index_peak = np.argmax(cumulative_returns[:index_max_drawdown+1])
        
        # Compute the maximum drawdown in percentage terms
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
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Calmar Ratio: {calmar_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")

        self.reset_portfolio()

        cumulative_returns_nr, strategy_returns_nr = self.backtest(regime_switching=False)
        sharpe_ratio_nr = self.calculate_sharpe_ratio(strategy_returns_nr)
        calmar_ratio_nr = self.calculate_calmar_ratio(strategy_returns_nr)
        sortino_ratio_nr = self.calculate_sortino_ratio(strategy_returns_nr)
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

        return cumulative_returns


# # Use yfinance to fetch data
# ticker = "^SPGSCI"
# start_date = "1991-05-01"
# interval = "1mo"

# # ticker = "DBC"
# # start_date = "2006-02-03"

# commodity = financial_data(ticker, start_date, interval)

# tickers = ['SPY', 'TLT', '^SBBMGLU'] 
# risk_parity_portfolio = RiskParityPortfolio(commodity, tickers)
# risk_parity_portfolio.run()
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

# Define the start date and interval for fetching financial data
interval = "1mo"

# Assuming you have a predefined start_date that is early enough to include all tickers
global_start_date = "1990-01-01" 
latest_inception_date = find_latest_inception_date(portfolios, commodity_tickers, global_start_date, interval)

# Loop through each commodity ticker
for commodity_ticker in commodity_tickers:
    start_date = "1991-05-01" if commodity_ticker == '^SPGSCI' else "2006-02-03"
    # Fetch financial data for the commodity ticker
    commodity_data = financial_data(commodity_ticker, start_date, interval)
    
    # Loop through each portfolio
    for tickers in portfolios:
        print(f"Running for Commodity Ticker: {commodity_ticker} with Portfolio: {tickers}")
        
        # Initialize the RiskParityPortfolio with the current commodity data and tickers
        risk_parity_portfolio = RiskParityPortfolio(commodity_data, tickers, latest_inception_date)
        
        # Run the analysis
        risk_parity_portfolio.run()
        
        # Optionally, reset the portfolio or perform additional cleanup here if necessary
        print("\nFinished running for the current configuration.\n")
