"""Module detecting regime through trend filtering."""
# import sys
# sys.path.append('/Users/elliebae/Documents/Senior Thesis/senior-thesis-1/senior-thesis')
import scipy
import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from itertools import groupby
from exploratory_data_analysis import financial_data

class regime_detection_tf:
    def __init__(self, financial_data, lamda_vals) -> None:
        self.df = financial_data.preprocess_data()
        self.financial_data = financial_data
        self.n = self.df.shape[0]
        self.lamda_vals = lamda_vals
        self.Y = self.add_features(self.df)
        self.df_tr = self.df[:int(0.8 * len(self.Y))]

    def add_features(self, data):
        features = pd.DataFrame(index=data.index)
        features['observation'] = data['Cumulative Log Return']

        return features

    def split_data(self, train_size=0.8):
        total_size = len(self.Y)
        train_end = int(train_size * total_size)
        self.train_data = self.Y.iloc[:train_end]
        self.test_data = self.Y.iloc[train_end:]

    def trend_filtering(self, lambd, time_series):
        time_series = np.array(time_series).flatten()  
        n = len(time_series)
        x = cp.Variable(n)  
        
        # Second-order difference matrix
        D = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()
        
        # Define the objective function
        obj_fn = 1/2 * cp.sum_squares(time_series - x) + lambd * cp.norm(D @ x, 1)
        objective = cp.Minimize(obj_fn)
        problem = cp.Problem(objective)
        problem.solve()
        prices_smoothed = x.value
        
        return prices_smoothed


    def trend_filtering_tuning(self, time_series, p_c):
        y = time_series.values
        n = len(y)
        I_n = np.eye(n)
        D = np.diff(I_n, 2).T
        psi = (2 * np.sin(np.pi / p_c)) ** (-4)
        x_hp = np.linalg.inv(I_n + psi * D.T @ D) @ y

        y = y.flatten()
        x_hp = x_hp.flatten()

        # ℓ1 minimization
        x_ast = cp.Variable(n)
        objective = cp.Minimize(cp.norm(D @ x_ast, 1))

        constraints = [cp.sum_squares(y - x_ast) <= cp.sum_squares(y - x_hp)]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        lambda_val = 2 * np.linalg.norm(np.linalg.inv(D @ D.T) @ (D @ (y - x_ast.value)), np.inf)

        # ℓ1 trend filtering
        x_lt = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(y - x_lt) + lambda_val * cp.norm(D @ x_lt, 1))
        prob = cp.Problem(objective)
        prob.solve()

        return lambda_val, x_lt.value, x_ast.value
    
    def lambda_cv(self):
        smoothed_prices_lamdas = []
        n_splits = 10
        best_lambda = None
        best_difference = -float('inf')
        tscv = TimeSeriesSplit(n_splits=n_splits)

        smoothed_prices_lamdas = []
        for lambd in self.lamda_vals:
            smoothed_prices = self.trend_filtering(lambd, self.Y)
            smoothed_prices_lamdas.append(smoothed_prices)

        smoothed_df = pd.DataFrame(smoothed_prices_lamdas)
        smoothed_df.index = self.lamda_vals
        smoothed_df.columns = self.df.index

        for i, lambd in enumerate(self.lamda_vals):
            differences = []
            smoothed_prices = smoothed_df.iloc[i]

            for train_index, test_index in tscv.split(smoothed_prices):
                smoothed_train = smoothed_prices[train_index]
                # Determine regimes from smoothed series
                regimes = np.where(np.array(smoothed_train[1:].values - smoothed_train[:-1].values) > 0, 'E', 'C')
                # Calculate next period returns
                next_returns = np.roll(smoothed_train.values, -1)[:-1]  # Drop the last value as there's no "next" return for it
                # Identify regime shifts
                shifts = np.where(regimes[:-1] != np.roll(regimes, -1)[:-1])[0]

                consecutive_crash_returns = []

                for idx in shifts:  
                    if regimes[idx] == 'E' and regimes[idx+1] == 'C':
                        end_of_crash = shifts[np.where(shifts > idx)[0][0]] if any(shifts > idx) else len(regimes) - 1
                        consecutive_crash_returns.extend(next_returns[idx+1:end_of_crash+1])

                if len(consecutive_crash_returns) > 0:
                    difference = np.mean(next_returns[regimes == 'E']) - np.mean(consecutive_crash_returns)
                    differences.append(difference)

            avg_difference = np.mean(differences)

            if avg_difference > best_difference:
                best_difference = avg_difference
                best_lambda = lambd
            print('best lambda', best_lambda)
        return best_lambda, smoothed_prices_lamdas
    
    # def plot_lambdas(self):
    #     _, smoothed_prices_lamdas = self.lambda_cv()
    #     cols = int(len(self.lamda_vals)/2)
    #     figs, axs = plt.subplots(2, cols)
    #     figs.set_figwidth(16)
    #     figs.set_figheight(10)
    #     figs.suptitle('The Smoothed Log-Returns of DBC Obtained Through Trend Filtering',  fontsize=12, y=0.93)

    #     for i in range(2):
    #         for j in range(cols):
    #             axs[i,j].plot(self.df.index, self.Y, label="Observed log-returns")
    #             axs[i,j].plot(self.df.index, smoothed_prices_lamdas[ 2*i + j ], label="Smoothed log-returns")
    #             axs[i,j].legend()
    #             axs[i,j].set_xlabel('Year', fontsize=12)
    #             axs[i,j].set_title('$\lambda$=%.2f' %self.lamda_vals[ 2*i + j ])
    #             axs[i,j].tick_params(axis='x', labelrotation = 90, labelsize=8)

    #     plt.show()

    #     return None
    
    
    def plot_lambdas(self):
        _, smoothed_prices_lamdas = self.lambda_cv()
        num_lambdas = len(self.lamda_vals)
        rows = num_lambdas // 4 + (num_lambdas % 4 > 0)
        cols = 4  
        fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 5))
        fig.suptitle('The Smoothed Log-Returns of GSCI Obtained Through Trend Filtering', fontsize=25, y=0.93)

        for i, (lambd, smoothed_prices) in enumerate(zip(self.lamda_vals, smoothed_prices_lamdas)):
            row, col = divmod(i, cols)
            if rows > 1:
                ax = axs[row, col]
            else:
                ax = axs[col]
            ax.plot(self.df.index, self.Y['observation'], label="Observed log-returns")  # Assuming 'observation' holds the log-returns
            ax.plot(self.df.index, smoothed_prices, label="Smoothed log-returns", linestyle='--')
            ax.legend()
            ax.set_xlabel('Year', fontsize=16)
            ax.set_title(f'$\lambda$={lambd:.2f}', fontsize=20)
            ax.tick_params(axis='x', labelrotation=90, labelsize=10)

        if num_lambdas % cols != 0:
            for j in range(i + 1, rows * cols):
                fig.delaxes(axs.flatten()[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        return None

    
    def run_model(self, tuning = False):
        if tuning:
            best_lambda, best_smoothed_prices, _ = self.trend_filtering_tuning(self.Y, 40)
        else:
            best_lambda, _ = self.lambda_cv()
            best_smoothed_prices = self.trend_filtering(best_lambda, self.Y)
       
        # best_smoothed_prices = np.insert(best_smoothed_prices, 0, 0.0)

        regimes = np.where(np.array(best_smoothed_prices[1:] - best_smoothed_prices[:-1]) > 0, 'C', 'E')
        # count_dups = [sum(1 for _ in group) for _, group in groupby(regime_sample)]
        # regime_seq = [r for r, group in groupby(regime_sample)]

        regimes = np.insert(regimes,0, regimes[0])
        self.df['Smoothed'] = best_smoothed_prices
        self.df['regimes'] = regimes

        # periods_e = df[df['Regime']=='E'].index
        # periods_c = df[df['Regime']=='C'].index

        # regime_map = dict(map(lambda i,j : (i,j) , df.index, regime_sample))
        regime_shifts = self.df.iloc[np.where(regimes[:-1] != np.roll(regimes, -1)[:-1])[0]]

        return regime_shifts
    
    def plot_best_lambda(self, tuning=True):
        regime_shifts = self.run_model(tuning=tuning)
        fig, ax = plt.subplots(dpi=300, figsize =(10,6))
        ax.plot(self.df.index, self.df['Cumulative Log Return'], label="Observed log-returns")
        ax.plot(self.df.index, self.df.Smoothed, label="Smoothed log-returns")

        for i, idx in enumerate(regime_shifts.index[:-1]):
            if regime_shifts.iloc[i].regimes == 'C':
                ax.axvspan(idx, regime_shifts.index[i+1], color="green", alpha=0.3)
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Log Return")
        ax.set_title("Contraction Regimes of DBC Identified Through Trend Filtering")
        ax.legend()

        plt.show()
        
        return None



# ticker = "^SPGSCI"
# start_date = "1991-05-01"
# interval = "1mo"

# # ticker = "DBC"
# # start_date = "2006-02-03"

# gsci = financial_data(ticker, start_date, interval)
# lambda_vals = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
# model_gsci = regime_detection_tf(gsci, lambda_vals)
# # model_gsci.run_model(tuning=True)
# model_gsci.plot_lambdas()
