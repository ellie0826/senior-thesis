import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from regime_detection_tf import regime_detection_tf
from regime_prediction import regime_prediction_ml
from exploratory_data_analysis import financial_data

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class RiskParityPortfolio:
    def __init__(self, financial_data, tickers, prediction_model):
        """
        :param asset_returns: DataFrame containing the historical returns of assets.
        :param regime_shifts: DataFrame containing the regime shifts.
        """
        self.financial_data = financial_data
        self.tickers = tickers
        self.asset_returns = self.fetch_returns()
        self.prediction_model = prediction_model
        self.regime_shifts = self.detect_regime_shifts()
        self.weights = {}
        self.w_n = len(tickers)+1

    def fetch_returns(self):
        other_data = yf.download(self.tickers, start=self.financial_data.start_date, end=self.financial_data.end_date, interval=self.financial_data.interval)['Adj Close']
        other_returns = other_data.pct_change().dropna()
        # other_returns.index = pd.to_datetime(other_returns.index, format="%Y-%m-%d").round('D')

        primary_returns = self.financial_data.preprocess_data()
        primary_returns.index = pd.to_datetime(primary_returns.index, format="%Y-%m-%d").round('D')
        primary_returns = primary_returns['Daily Return']

        combined_returns = pd.concat([primary_returns, other_returns], axis=1).dropna()

        return combined_returns
    
    def calculate_covariance_for_regime(self, returns, regime, current_date):
        # Filter returns for dates corresponding to the given regime up to the current date
        regime_dates = self.prediction_model.final_df[(self.prediction_model.final_df['predicted regime label'] == regime) & (self.prediction_model.final_df.index < current_date)].index
        filtered_returns = returns.loc[regime_dates]
        return filtered_returns.cov() * 12  # Annualized covariance matrix
    
    def calculate_weighted_covariance_matrix(self, current_date):
        crash_probability = self.prediction_model.final_df.loc[current_date, 'final probabilities']
        cov_matrix_normal = self.calculate_covariance_for_regime(self.asset_returns, 'E', current_date)
        cov_matrix_crash = self.calculate_covariance_for_regime(self.asset_returns, 'C', current_date)
        weighted_cov_matrix = (1 - crash_probability) * cov_matrix_normal + crash_probability * cov_matrix_crash
        return weighted_cov_matrix
    
    def calculate_covariance_matrix(self, returns):
            """
            Calculates the covariance matrix for the given returns.
            """
            cov_matrix = returns.cov().values * 12
            return cov_matrix
    
    def calculate_pf_volatility(self, weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def risk_contribution(self, weights, cov_matrix):
        portfolio_vol = self.calculate_pf_volatility(weights, cov_matrix)
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = np.multiply(weights, marginal_contrib) / portfolio_vol
        return risk_contrib
    
    def risk_parity_objective(self, weights, cov_matrix):
        target_risk_contrib = np.ones(len(weights)) / len(weights)
        actual_risk_contrib = self.risk_contribution(weights, cov_matrix)
        return np.sum((actual_risk_contrib - target_risk_contrib)**2)
    
    def optimize_weights(self, returns, weighted=False, current_date=None):
        """
        Optimizes the portfolio weights for risk parity using the covariance matrix.
        """
        print('returns in optimized weights', returns)
        initial_weights = np.ones(returns.shape[1]) / returns.shape[1]
        if weighted:
            cov_matrix = self.calculate_weighted_covariance_matrix(returns, current_date)
        else:
            cov_matrix = self.calculate_covariance_matrix(returns)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(returns.shape[1]))

        # Optimization
        opt_result = minimize(fun=self.risk_parity_objective,
                            x0=initial_weights,
                            args=(cov_matrix),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)

        # Optimized weights
        optimized_weights = opt_result.x
        print('weights', optimized_weights)
        return optimized_weights

    def detect_regime_shifts(self):
        """
        Detects regime shifts based on changes in the 'final regimes' column.
        """
        final_df = self.prediction_model.final_df.reset_index()
        final_df['shifts'] = final_df['final regimes'].diff().ne(0)
        regime_shifts = final_df[final_df['shifts']].index
        final_df.drop(columns=['shifts'], inplace=True)

        return regime_shifts
       
    def rebalance_portfolio_regime(self):
        idx_list = self.prediction_model.final_df.index.tolist()
        print('regime shifts', self.regime_shifts)
        for i in range(len(self.regime_shifts)-1):
            start, end = self.regime_shifts[i], self.regime_shifts[i+1]
            print('start', start, 'end', end)

            if idx_list[start] >= self.prediction_model.test_start_date:
                current_date = idx_list[start]
                print('current date', current_date)
                print('end date', idx_list[end])
                returns_subset = self.asset_returns[start:end]
                print('returns subset', returns_subset)
                weights = self.optimize_weights(returns_subset, weighted=True, current_date=current_date)
                self.weights[f'{start}_{end}'] = weights

    def rebalance_portfolio_quarterly(self):
        start_date = self.asset_returns.index.min()
        end_date = self.asset_returns.index.max()
        quarterly_dates = pd.date_range(start=start_date, end=end_date, freq='QS')

        if end_date not in quarterly_dates:
            quarterly_dates = quarterly_dates.union([end_date])

        quarterly_indices = [self.asset_returns.index.get_loc(date) for date in quarterly_dates]

        for i in range(len(quarterly_indices) - 1):
            start_idx, end_idx = quarterly_indices[i], quarterly_indices[i + 1]  
            if quarterly_dates[i] >= self.prediction_model.test_start_date:
                returns_subset = self.asset_returns[start_idx:end_idx]
                if not returns_subset.empty:
                    optimized_weights = self.optimize_weights(returns_subset)
                    # Save the optimized weights with the start and end indices as the key
                    self.weights[f'{start_idx}_{end_idx}'] = optimized_weights

    def calculate_60_40_sharpe_ratio(self):
        """
        Calculates the Sharpe Ratio for a 60/40 SPY/TLT portfolio.
        """
        # Fetch returns for SPY and TLT
        spy_tlt_data = yf.download(['SPY', 'TLT'], start=self.financial_data.start_date, end=self.financial_data.end_date, interval=self.financial_data.interval)['Adj Close']
        spy_tlt_returns = spy_tlt_data.pct_change().dropna()

        # Calculate weighted returns for the 60/40 portfolio
        weights_60_40 = np.array([0.6, 0.4])
        portfolio_returns_60_40 = (spy_tlt_returns * weights_60_40).sum(axis=1)

        mean_return_60_40 = portfolio_returns_60_40.mean()
        std_deviation_60_40 = portfolio_returns_60_40.std()
        sharpe_ratio_60_40 = mean_return_60_40 / std_deviation_60_40

        return sharpe_ratio_60_40

    def calculate_sharpe_ratio(self):
        """
        Evaluates the performance of the rebalanced portfolio using the Sharpe Ratio.
        """
        portfolio_returns = []
        for key, weights in self.weights.items():
            start, end = key.split('_')
            start, end = int(start), int(end)
            returns_subset = self.asset_returns.iloc[start:end]
            weighted_returns = (returns_subset * weights).sum(axis=1)
            portfolio_returns.extend(weighted_returns)

        mean_return = np.mean(portfolio_returns)
        std_deviation = np.std(portfolio_returns)
        epsilon = 1e-8  # Small constant to avoid division by zero
        sharpe_ratio = mean_return / (std_deviation + epsilon)

        return sharpe_ratio

    # def run(self):
    #     """
    #     Executes the portfolio construction and evaluation process.
    #     """
    #     self.rebalance_portfolio_quarterly()
    #     sharpe_ratio = self.calculate_sharpe_ratio()
    #     print(f'Sharpe Ratio without Regime-Switching: {sharpe_ratio}')
    #     self.rebalance_portfolio_regime()
    #     sharpe_ratio_regime = self.calculate_sharpe_ratio()
    #     print(f'Sharpe Ratio: {sharpe_ratio_regime}')
    #     sharpe_60_40 = self.calculate_60_40_sharpe_ratio()
    #     print(f"60/40 Portfolio Sharpe Ratio: {sharpe_60_40}")


    #     return sharpe_ratio

    def backtest(self):
        weights_record = {}
        for shift_date in self.regime_shifts:
            cov_matrix = self.calculate_weighted_covariance_matrix(shift_date)
            weights = self.optimize_weights(cov_matrix)
            weights_record[shift_date] = weights
        return weights_record

    def run(self):
        weights_record = self.backtest()
        # Further evaluation or calculation based on weights_record
        return weights_record
   
# Use yfinance to fetch data
ticker = "^SPGSCI"
start_date = "1991-05-01"
end_date = "2001-05-01"
interval = "1mo"

# dbc_ticker = "DBC"
# start_date_dbc = "2006-02-03"

gsci = financial_data(ticker, start_date, interval)
lambda_vals = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
model_gsci = regime_detection_tf(gsci, lambda_vals)
model_gsci.run_model(tuning=True)
# model_gsci.split_data(train_size=0.8)

macro_data_path = "current.csv"
model_type = 'logistic_regression'
gsci_prediction = regime_prediction_ml(model_gsci, macro_data_path, model_type=model_type, use_macro_data=True)
gsci_prediction.run()

tickers = ['SPY', 'TLT']  
risk_parity = RiskParityPortfolio(gsci, tickers, gsci_prediction)
risk_parity.run()
