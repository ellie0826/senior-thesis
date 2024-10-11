import pandas as pd
import numpy as np
import yfinance as yf
from regime_detection_tf import regime_detection_tf
from exploratory_data_analysis import financial_data
from regime_prediction import regime_prediction_ml
from scipy.optimize import minimize

class RiskParityPortfolio:
    def __init__(self, financial_data, tickers, prediction_model):
        self.financial_data = financial_data
        self.tickers = tickers
        self.prediction_model = prediction_model
        self.asset_returns = self.fetch_returns()
        # Calculate regime shifts for rebalancing
        self.regime_shift_dates = self.calculate_regime_shifts()

    def fetch_returns(self):
        other_data = yf.download(self.tickers, start=self.financial_data.start_date, end=self.financial_data.end_date, interval=self.financial_data.interval)['Adj Close']
        other_returns = other_data.pct_change().dropna()
        # other_returns.index = pd.to_datetime(other_returns.index, format="%Y-%m-%d").round('D')

        primary_returns = self.financial_data.preprocess_data()
        primary_returns.index = pd.to_datetime(primary_returns.index, format="%Y-%m-%d").round('D')
        primary_returns = primary_returns['Daily Return']

        combined_returns = pd.concat([primary_returns, other_returns], axis=1).dropna()
        return combined_returns
    
    def calculate_regime_shifts(self):
        """
        Identifies the dates where regime shifts occurred for rebalancing, returning date indices.
        """
        # final_df = self.prediction_model.final_df.reset_index()
        regime_changes = self.prediction_model.final_df['final regimes'].diff().ne(0)
        # Filter to get the dates of those changes
        # regime_shifts = final_df[regime_changes].index
        regime_shift_dates = self.prediction_model.final_df.index[regime_changes]
        
        # Further filter to select only those after the test_start_date
        # Ensure both regime_shift_dates and test_start_date are compatible datetime objects for comparison
        regime_shift_dates_filtered = regime_shift_dates[regime_shift_dates >= pd.to_datetime(self.prediction_model.test_start_date)]
        
        return regime_shift_dates_filtered.tolist()

    def calculate_covariance_for_regime(self, returns, regime, current_date):
        # Example: Normalize your DataFrame index to ensure it's at the desired frequency
        self.asset_returns.index = pd.to_datetime(self.asset_returns.index, format="%Y-%m-%d").round('D')
        regime_dates = self.prediction_model.final_df[(self.prediction_model.final_df['predicted regime label'] == regime) & (self.prediction_model.final_df.index < current_date)].index
        # Normalize the regime dates as well, assuming they're already Timestamp objects
        regime_dates = pd.to_datetime(regime_dates, format="%Y-%m-%d").round('D')
        print('regime dates', regime_dates)
        print('returns', returns)
        
        filtered_returns = returns[regime_dates]
        return filtered_returns.cov() * 12  # Annualized covariance matrix
    
    def calculate_weighted_covariance_matrix(self, current_date):
        crash_probability = self.prediction_model.final_df.loc[current_date, 'final probabilities']
        cov_matrix_normal = self.calculate_covariance_for_regime(self.asset_returns, 'E', current_date)
        cov_matrix_crash = self.calculate_covariance_for_regime(self.asset_returns, 'C', current_date)
        weighted_cov_matrix = (1 - crash_probability) * cov_matrix_normal + crash_probability * cov_matrix_crash
        return weighted_cov_matrix
    
    def risk_parity_objective(self, weights, args):
        cov_matrix = args[0]
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contributions = cov_matrix @ weights
        risk_contributions = weights * marginal_contributions / portfolio_volatility
        target_risk_contribution = np.ones(len(weights)) / len(weights)
        return np.sum((risk_contributions - target_risk_contribution)**2)
    
    def optimize_weights(self, cov_matrix):
        n_assets = self.asset_returns.shape[1]
        initial_weights = np.ones(n_assets) / n_assets
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        result = minimize(self.risk_parity_objective, initial_weights, args=[cov_matrix], method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization Failed")
        
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
                
    def backtest(self):
        weights_record = {}
        for shift_date in self.regime_shift_dates:
            cov_matrix = self.calculate_weighted_covariance_matrix(shift_date)
            print('cov', cov_matrix)
            weights = self.optimize_weights(cov_matrix)
            print('weights', weights)
            weights_record[shift_date] = weights
        return weights_record

    def run(self):
        weights_record = self.backtest()
        # Further evaluation or calculation based on weights_record
        return weights_record


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
