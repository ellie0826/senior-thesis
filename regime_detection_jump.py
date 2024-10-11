"""Module detecting regime with continuous statistical jump models."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from exploratory_data_analysis import financial_data

class regime_detection_jump:
    def __init__(self, financial_data, K, lambd):
        self.financial_data = financial_data
        self.df = financial_data.preprocess_data()  # Preprocessed financial data
        self.K = K       # Number of states
        self.lambd = lambd  # Jump penalty parameter
        self.Y = self.add_features(self.df)

    def add_features(self, data):
        features = pd.DataFrame(index=data.index)
        features['observation'] = data['Daily Return']

        # # Calculate rolling averages for 3, 6, 9, and 12 months for additional features
        # for months in [3, 6, 9, 12]:
        #     rolling_avg = data['Daily Return'].rolling(window=months).mean()
        #     features[f'{months}m_rolling_avg'] = rolling_avg.fillna(method='bfill')

        return features

    def split_data(self, train_size=0.6):
        total_size = len(self.Y)
        train_end = int(train_size * total_size)
        self.train_data = self.Y.iloc[:train_end]
        self.test_data = self.Y.iloc[train_end:]

    def fit_model_parameters(self, S, Y):
        m = gp.Model("fit_model_parameters")
        Theta = m.addVars(self.K, Y.shape[1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Theta")
        T, K = Y.shape[0], self.K

        # Manually constructing the loss term
        loss = 0
        for t in range(T):
            for k in range(K):
                for j in range(Y.shape[1]):
                    loss += S[t, k] * (Y.iloc[t, j] - Theta[k, j]) ** 2

        # Set the objective function
        m.setObjective(loss, GRB.MINIMIZE)
        m.optimize()

        return np.array([[Theta[k, j].X for j in range(Y.shape[1])] for k in range(self.K)])

    def fit_state_sequence(self, Theta, Y):
        m = gp.Model("fit_state_sequence")
        T, K = Y.shape[0], self.K
        S = m.addVars(T, K, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="S")

        # Constructing loss matrix
        L = {(t, k): sum((Y.iloc[t, j] - Theta[k, j])**2 for j in range(Y.shape[1]))
             for t in range(T) for k in range(K)}

        # Calculate the inner product ⟨S, L⟩
        inner_product = sum(S[t, k] * L[t, k] for t in range(T) for k in range(K))

        # Calculate the jump penalty
        jump_penalty = sum((S[t, k] - S[t - 1, k])**2 for t in range(1, T) for k in range(K))

        # Set the objective function
        m.setObjective(inner_product + self.lambd / 4.0 * jump_penalty, GRB.MINIMIZE)

        # Constraints
        for t in range(T):
            m.addConstr(sum(S[t, k] for k in range(K)) == 1, f"sum_probability_{t}")

        m.optimize()

        S = np.array([[S[t, k].X for k in range(K)] for t in range(T)])

        return S


    def coordinate_descent(self, Y, max_iter=100, tolerance=1e-5, S_initial=None):
        if S_initial is None or S_initial.shape[0] != Y.shape[0]:
            S = np.random.rand(Y.shape[0], self.K)  # Random initialization of S
        else:
            S = S_initial  # Use provided initial state sequence
        Theta = np.zeros((self.K, Y.shape[1]))  # Initialize Theta

        for iteration in range(max_iter):
            Theta_old, S_old = Theta.copy(), S.copy()

            # Update Theta given S
            Theta = self.fit_model_parameters(S, Y)

            # Update S given new Theta, ensuring updates are time-consistent
            S = self.fit_state_sequence(Theta, Y)

            # Check for convergence
            if np.linalg.norm(S - S_old) < tolerance and np.linalg.norm(Theta - Theta_old) < tolerance:
                print(f"Convergence reached after {iteration+1} iterations.")
                break

        return Theta, S


    def evaluate_model(self, Theta, Y):
        T, K = Y.shape[0], self.K
        _, S = self.coordinate_descent(Y)

        # Calculate loss
        inner_product = sum(S[t, k] * sum((Y.iloc[t, j] - Theta[k, j])**2 for j in range(Y.shape[1]))
                            for t in range(T) for k in range(K))
        jump_penalty = sum((S[t, k] - S[t - 1, k])**2 for t in range(1, T) for k in range(K))

        loss = inner_product + self.lambd / 4.0 * jump_penalty
        return loss

    def time_series_cross_validation(self, lambda_values, n_splits=5):
        """
        Perform time series cross-validation to find the optimal lambda value.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_lambda = None
        best_performance = float('inf')

        for lambda_val in lambda_values:

            total_performance = 0
            for train_index, val_index in tscv.split(self.train_data):
                train_data, val_data = self.Y.iloc[train_index], self.Y.iloc[val_index]
                self.lambd = lambda_val
                _, S_train = self.coordinate_descent(train_data)
                Theta, _ = self.coordinate_descent(val_data, S_initial=S_train)  # Use the last state sequence as initial for the next
                performance = self.evaluate_model(Theta, val_data)
                total_performance += performance

            avg_performance = total_performance / n_splits
            print(f'lambda: {lambda_val} with performance: {avg_performance}')
            if avg_performance < best_performance:
                best_performance = avg_performance
                best_lambda = lambda_val

        self.lambd = best_lambda
        print(f"Optimal lambda: {best_lambda} with performance: {best_performance}")

    def fit_full_model(self):
        self.full_Theta, self.full_S = self.coordinate_descent(self.Y)

    def test_model(self):
        Theta, _ = self.coordinate_descent(self.train_data)
        final_performance = self.evaluate_model(Theta, self.test_data)
        print(f"Final Model Performance on Test Data: {final_performance}")

    
    def run_model(self):
        self.fit_full_model()
        S = np.array(self.full_S)
        regimes = np.argmax(S, axis=1)
        
        # Calculate average return for each regime
        avg_returns = {k: self.df.loc[regimes == k, 'Daily Return'].mean() for k in range(self.K)}
        
        # Determine regimes
        crash_label, expansive_label = (0, 1) if avg_returns[0] < avg_returns[1] else (1, 0)
        
        # Map regimes to labels
        regime_labels = ['E' if r == crash_label else 'C' for r in regimes]
        
        self.df['regimes'] = regime_labels
        print('Regime labels:', regime_labels)
        
        regime_shifts = self.df.iloc[np.where(regimes[:-1] != np.roll(regimes, -1)[:-1])[0]]

        
        return regime_labels, regime_shifts

    # def run_model(self):
    #     # self.test_model()
    #     self.fit_full_model()
    #     S = np.array(self.full_S)
    #     regimes = np.argmax(S, axis=1)

    #     # Mapping 0 to 'C' for crash and 1 to 'E' for expansion
    #     regime_labels = ['C' if r == 0 else 'E' for r in regimes]

    #     self.df['regimes'] = regime_labels
    #     print('regime lables', regime_labels)

    #     regime_shifts = self.df.iloc[np.where(regimes[:-1] != np.roll(regimes, -1)[:-1])[0]]
    #     print(regime_shifts)

    #     return regime_labels, regime_shifts


    def predict_regimes(self, new_data):
        # Preprocess new_data to match the training data format
        preprocessed_data = self.add_features(new_data.preprocess_data())
        
        # Predict using the last state of S as the initial state for the new data
        _, S_new = self.coordinate_descent(preprocessed_data, S_initial=self.full_S[-1].reshape(1, self.K))
        
        # Convert S_new to a DataFrame 
        regime_probabilities = pd.DataFrame(S_new, columns=[f'Regime_{k}' for k in range(self.K)], index=preprocessed_data.index)
        
        return regime_probabilities
    
    def plot_regimes(self):
        _, regime_shifts = self.run_model()
        S = np.array(self.full_S)  # Probabilities for each regime
        
        fig1, ax1 = plt.subplots(dpi=300, figsize=(10, 6))
        ax1.plot(self.df.index, self.df['Cumulative Log Return'], label="Observed returns")

        if len(regime_shifts.index) > 0:
            # Add the start of the dataframe
            regime_shifts_with_start = pd.concat([self.df.iloc[[0]], regime_shifts])
            # Add the end of the dataframe
            regime_shifts_with_end = pd.concat([regime_shifts_with_start, self.df.iloc[[-1]]])

            # Iterate through the regime shift data points
            for (prev_idx, prev_row), (next_idx, next_row) in zip(regime_shifts_with_start.iterrows(), regime_shifts_with_end.iloc[1:].iterrows()):
                ax1.axvspan(prev_idx, next_idx, color='grey' if prev_row['regimes'] == 'C' else 'green', alpha=0.3)

        ax1.legend()
        plt.show()

    

    # def cross_validation(self, lambda_values, k=10):
    #     kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #     best_lambda = None
    #     best_performance = float('inf')

    #     for lambda_val in lambda_values:
    #         total_performance = 0
    #         for train_index, val_index in kf.split(self.train_data):
    #             train_data, val_data = self.train_data.iloc[train_index], self.train_data.iloc[val_index]
    #             self.lambd = lambda_val
    #             Theta, _ = self.coordinate_descent(train_data)
    #             performance = self.evaluate_model(Theta, val_data)
    #             total_performance += performance

    #         avg_performance = total_performance / k
    #         print('lambd', lambda_val, 'performance', avg_performance)
    #         if avg_performance < best_performance:
    #             best_performance = avg_performance
    #             best_lambda = lambda_val
    #     self.lambd = best_lambda
    #     print('best lambd', best_lambda)



    def plot_regimes_probability(self):
        self.run_model()
        S = np.array(self.full_S)  # Probabilities for each regime

        fig1, ax1 = plt.subplots(dpi=300, figsize=(10, 6))
        observed_returns_line, = ax1.plot(self.df.index, self.df['Cumulative Log Return'], label="Observed returns", color="red", linestyle='-')

        ax1.set_xlabel("Year")
        ax1.set_ylabel("Cumulative Log Return")
        ax2 = ax1.twinx()
        # Plot for probabilities
        contraction_probability_line, = ax2.plot(self.df.index, S[:, 0], label="Contraction Probability", color="blue", linestyle='--')
        # ax3.plot(self.df.index, S[:, 1], label="Growth Probability", alpha=0.4)
        ax2.set_ylabel("Probability")
        ax2.set_title('Probabilities of Contraction Regime Identified from DBC')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

        lines = [observed_returns_line, contraction_probability_line]
        labels = [line.get_label() for line in lines]

            # Place a single legend on the figure
        ax2.legend(lines, labels, loc='upper left')

        plt.show()

        return None
    
# ticker = "^SPGSCI"
# start_date = "1991-05-01"
interval = "1mo"


ticker = "DBC"
start_date = "2006-02-03"

gsci = financial_data(ticker, start_date, interval=interval)
model = regime_detection_jump(gsci, K=2, lambd=0.05)
# model.split_data()

# lambda_values = [0.001, 0.004, 0.01, 0.03]
# # # model.time_series_cross_validation(lambda_values=lambda_values)
# model.run_model()
model.plot_regimes_probability()
# start_date2 = "2001-05-02"
# gsci_new = financial_data(ticker, start_date2, interval)
# regime_probs = model.predict_regimes(gsci_new)
# print(regime_probs)


# model.test_model()