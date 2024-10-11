import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
from regime_detection_tf import regime_detection_tf
from regime_prediction import regime_prediction_ml
from exploratory_data_analysis import financial_data
import pickle

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os


class RiskParityPortfolio:
    def __init__(self, financial_data, tickers, model_output_filename='model_output.pkl'):
        self.financial_data = financial_data
        self.tickers = tickers
        self.asset_returns = self.fetch_returns()
        
        # Attempt to load the prediction model's output if available
        if os.path.exists(model_output_filename):
            with open(model_output_filename, 'rb') as file:
                self.prediction_model = pickle.load(file)
        else:
            # Load or define your prediction model here if not loading from file
            # For example:
            self.prediction_model = self.load_prediction_model()
            self.prediction_model.run()  # Assuming your model has a .run() method to generate outputs
            with open(model_output_filename, 'wb') as file:
                pickle.dump(self.prediction_model, file)
                
        self.regime_shifts = self.calculate_regime_shifts()
        self.weights = {}
        self.w_n = len(tickers)+1


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
        # Shift the 'final regimes' column down by one to compare with the previous row
        shifted_regimes = self.prediction_model.final_df['predicted regime label'].shift(1)
        
        # Compare the original column with the shifted version to find where the value changes
        regime_changes = self.prediction_model.final_df['predicted regime label'] != shifted_regimes
        regime_shift_dates = self.prediction_model.final_df.index[regime_changes]
        regime_shift_dates = pd.to_datetime(regime_shift_dates, format="%Y-%m-%d").round('D')
        
        regime_shift_dates_filtered = regime_shift_dates[regime_shift_dates >= pd.to_datetime(self.prediction_model.test_start_date)]
        
        regime_shift_dates_formatted = [date.strftime('%Y-%m-%d') for date in regime_shift_dates_filtered]
        
        return regime_shift_dates_formatted

    
    def calculate_covariance_for_regime(self, returns, regime, current_date):
        # Filter returns for dates corresponding to the given regime up to the current date
        regime_dates = self.prediction_model.final_df[(self.prediction_model.final_df['predicted regime label'] == regime) & (self.prediction_model.final_df.index < current_date)].index
        regime_dates = regime_dates[regime_dates >= pd.to_datetime(self.prediction_model.test_start_date)]
        regime_dates = [date.strftime('%Y-%m-%d') for date in regime_dates]

        print('regime', regime)
        print('returns', returns)
        print('current date', current_date)
        print('regime dates', regime_dates)

        filtered_returns = returns.loc[regime_dates]
        print('regime', regime, 'regimie cov', filtered_returns.cov()*12)

        return filtered_returns.cov() * 12  # Annualized covariance matrix
    
    def calculate_weighted_covariance_matrix(self, current_date):
        crash_probability = self.prediction_model.final_df.loc[current_date, 'final probabilities']
        cov_matrix_normal = self.calculate_covariance_for_regime(self.asset_returns, 'E', current_date)
        cov_matrix_crash = self.calculate_covariance_for_regime(self.asset_returns, 'C', current_date)
        weighted_cov_matrix = (1 - crash_probability) * cov_matrix_normal + crash_probability * cov_matrix_crash
        print('weighted cov', weighted_cov_matrix)        
        
        return weighted_cov_matrix
    
    def calculate_risk_parity_weights(self, cov_matrix):
        """
        Calculate portfolio weights based on risk parity approach.
        
        :param cov_matrix: Covariance matrix of the asset returns
        :return: Optimized portfolio weights
        """
        # Number of assets
        n = cov_matrix.shape[0]
        initial_weights = np.ones(n) / n
        # Objective function: minimize the sum of squared differences between 
        # each asset's risk contribution and the total risk divided by number of assets

        def calculate_pf_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
        def risk_contribution(weights):
            portfolio_vol = calculate_pf_volatility(weights)
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = np.multiply(weights, marginal_contrib) / portfolio_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            target_risk_contrib = np.ones(len(weights)) / len(weights)
            actual_risk_contrib = risk_contribution(weights)
            return np.sum((actual_risk_contrib - target_risk_contrib)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))

        # Optimization
        opt_result = minimize(fun=risk_parity_objective,
                            x0 = initial_weights,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)

        # Optimized weights
        optimized_weights = opt_result.x
        print('weights', optimized_weights)
       
        if not opt_result.success:
            raise BaseException('Risk parity optimization did not converge.')
        
        return optimized_weights
    
    def backtest(self):
        weights_record = {}
        for shift_date in self.regime_shifts:
            cov_matrix = self.calculate_weighted_covariance_matrix(shift_date)
            weights = self.calculate_risk_parity_weights(cov_matrix)
            print('weights', weights)
            # weights_record[shift_date] = weights
        return weights_record
  
# Use yfinance to fetch data
ticker = "^SPGSCI"
start_date = "1991-05-01"
interval = "1mo"

gsci = financial_data(ticker, start_date, interval)

tickers = ['SPY', 'TLT']  
risk_parity = RiskParityPortfolio(gsci, tickers)
# print(risk_parity.asset_returns)
# print(risk_parity.regime_shifts)
print(risk_parity.prediction_model.final_df[risk_parity.prediction_model.final_df['predicted regime label']=='E'])
print(risk_parity.prediction_model.final_df[risk_parity.prediction_model.final_df['predicted regime label']=='C'])
risk_parity.backtest()