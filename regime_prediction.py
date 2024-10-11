import pandas as pd
import numpy as np
from regime_detection_tf import regime_detection_tf
from exploratory_data_analysis import financial_data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import LSTM, Dropout
import matplotlib.pyplot as plt



class regime_prediction_ml:
    def __init__(self, regime_model, macro_data_path, model_type='logistic_regression', use_macro_data=True):
        self.regime_model= regime_model
        self.macro_data_path = macro_data_path
        self.use_macro_data = use_macro_data
        if self.use_macro_data:
            self.macro_data = pd.read_csv(self.macro_data_path)
        else:
            self.macro_data = pd.DataFrame()  # Empty DataFrame if not using macro data
        self.model = None
        self.model_type = model_type
        self.target_column = 'regimes'
        
    def load_and_merge_data(self):
        macro_data = self.macro_data.rename(columns={'sasdate': 'Date'})
        macro_data = macro_data[macro_data['Date'].str.match(r'^\d{1,2}/\d{1,2}/\d{4}$')]
        macro_data['Date'] = pd.to_datetime(macro_data['Date'], format='%m/%d/%Y')

        financial_data_reset = self.regime_model.df.reset_index()

        merged_data = pd.merge(financial_data_reset, macro_data, on='Date', how='inner')
        merged_data.ffill(inplace=True)
        merged_data.dropna(inplace=True)

        if 'regimes' not in merged_data.columns:
            raise KeyError("The 'regimes' column not found in the merged dataset")
        
        merged_data.index = pd.to_datetime(merged_data['Date'], format="%Y-%m-%d")
        columns_to_keep = ['regimes', 'Daily Return'] + [col for col in macro_data.columns if col != 'Date']
        merged_data_filtered = merged_data[columns_to_keep]

        return merged_data_filtered
    
    
    def preprocess_data(self, data):
        
        features = data.drop(self.target_column, axis=1)
        target = data[self.target_column]

        # Impute missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        features_imputed = imputer.fit_transform(features)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_imputed)

        # Convert target to categorical if not numeric type
        if not np.issubdtype(target.dtype, np.number):
            target = pd.get_dummies(target, drop_first=True)
        
        train_idx = round(len(target.values) * 0.4)
        X_train, X_test = features_scaled[:train_idx], features_scaled[train_idx:]
        y_train, y_test = target[:train_idx]['E'], target[train_idx:]['E']

        test_start_date = self.regime_model.df.index[train_idx]
        test_start_date = pd.to_datetime(test_start_date, format="%Y-%m-%d")
        self.test_start_date = test_start_date
        
        return train_idx, X_train, X_test, y_train, y_test

    def create_model(self, input_shape):
        if self.model_type == 'nn':
            print("Creating NN model...")
            self.model = Sequential([
                Dense(64, activation='relu', input_shape=(input_shape,)),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')  
            ])
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier()
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(bootstrap = True, max_depth = None, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 100)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
        elif self.model_type == 'rnn':
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=(input_shape, 1)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        else:
            raise ValueError(f"Model type '{self.model_type}' not supported.")

    def train_model(self, X_train, y_train):
        if self.model_type == 'nn':
            self.model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32)
        elif self.model_type == 'rnn':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
            y_train = np.reshape(y_train, (y_train.shape[0],1))
            self.model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32)
        else:
            self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        if self.model_type in ['nn', 'rnn']:
            if self.model_type == 'rnn':
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                y_test = np.reshape(y_test, (y_test.shape[0], 1))
            
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
            probabilities = self.model.predict(X_test)

            # Convert probabilities to binary outcomes
            if probabilities.shape[-1] > 1:
                # For multi-class classification
                y_pred = probabilities.argmax(axis=-1)
            else:
                # For binary classification
                y_pred = (probabilities > 0.5).astype("int32").reshape(-1)
                
        else:
            y_pred = self.model.predict(X_test)
            probabilities = self.model.predict_proba(X_test)[:, 1]  
            test_accuracy = accuracy_score(y_test, y_pred)

        return probabilities, y_pred


    
    def iterative_training_and_prediction(self, X_train_initial, y_train_initial, X_test, y_test, step_size=4):
        X_train = np.copy(X_train_initial)
        y_train = np.copy(y_train_initial)

        probabilities = []
        predictions = []

        i = 0
        while i < len(X_test):
            current_step_size = min(step_size, len(X_test) - i)
            self.train_model(X_train, y_train)

            next_probabilities, next_predictions = self.evaluate_model(X_test[i:i+current_step_size], y_test[i:i+current_step_size])
            probabilities.extend(next_probabilities)
            predictions.extend(next_predictions)

            X_train = np.vstack((X_train, X_test[i:i+current_step_size]))
            y_train = np.concatenate((y_train, y_test[i:i+current_step_size]))
            
            i += current_step_size
        
        # Train on all available data and predict the training set for probabilities
        self.train_model(X_train, y_train)
        training_probabilities, _ = self.evaluate_model(X_train_initial, y_train_initial)

        # Concatenate training probabilities with the iteratively predicted probabilities
        final_probabilities = np.concatenate([training_probabilities, probabilities])

        # Use actual training labels followed by the predicted labels for the test set
        final_pred = np.concatenate([y_train_initial, predictions])


        self.evaluate_predictions(predictions, y_test, probabilities)

        return final_probabilities, final_pred

    def evaluate_predictions(self, predictions, y_test, probabilities, plot=True):
        predictions = np.array(predictions)
        y_test = np.array(y_test)

        test_accuracy = accuracy_score(y_test, predictions > 0.5)
        print(f"Test Accuracy: {test_accuracy}")

        test_size = len(predictions)

        if plot:
            time_series_data = self.regime_model.df[-test_size:]['Cumulative Log Return']
            dates = np.array(time_series_data.index)

            fig, ax1 = plt.subplots(figsize=(14, 7))

            color = 'tab:grey'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Log Return', color=color)
            ln1 = ax1.plot(dates, time_series_data, label='Cumulative Log Return', color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('Probability', color=color)  
            ln2 = ax2.plot(dates, probabilities, label='Probability of Contraction Regime', color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

            def plot_contiguous_segments(dates, data, ax, color, label):
                start_idx = None
                label_assigned = False
                for i in range(len(data)):
                    if data[i] == 1 and start_idx is None:
                        start_idx = i 
                    elif data[i] == 0 and start_idx is not None:
                        ax.axvspan(dates[start_idx], dates[i], color=color, alpha=0.3, label=label if not label_assigned else "")
                        if not label_assigned:
                            label_assigned = True
                        start_idx = None
                if start_idx is not None:
                    ax.axvspan(dates[start_idx], dates[-1], color=color, alpha=0.3, label=label if not label_assigned else "")

            plot_contiguous_segments(dates, y_test, ax1, 'red', 'Actual Contraction Regime')  # Actual contraction regimes
            plot_contiguous_segments(dates, predictions > 0.5, ax1, 'blue', 'Predicted Contraction Regime')  # Predicted contraction regimes

            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')

            title = f'Contraction Regime Prediction vs. Actual Outcomes of {self.regime_model.financial_data.ticker} ({self.model_type})'
            plt.title(title)
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.show()



    def run(self):
        data = self.load_and_merge_data()
        train_idx, X_train, X_test, y_train, y_test = self.preprocess_data(data)

        self.create_model(input_shape=X_train.shape[1])
        final_probabilities, final_pred = self.iterative_training_and_prediction(X_train, y_train, X_test, y_test)
        final_regime = ['C' if pred else 'E' for pred in final_pred]

        data['predicted regime label'] = final_regime
        data['final probabilities'] = final_probabilities
        self.final_df = data

    def tune_hyperparameters(self):
        data = self.load_and_merge_data()
        _, X_train, _, y_train, _ = self.preprocess_data(data)
        self.create_model(input_shape=X_train.shape[1])

        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.001, 0.01, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            raise ValueError(f"Hyperparameter tuning not supported for model type '{self.model_type}'")


        # Initialize the TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits=5) 

        # Initialize the GridSearchCV object with TimeSeriesSplit
        grid_search = GridSearchCV(self.model, param_grid, cv=tscv, n_jobs=-1, verbose=2)

        # Perform the grid search on the training data
        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        # Print the best parameters and the best score achieved
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score achieved: ", grid_search.best_score_)

 
interval = "1mo"

ticker = "DBC"
start_date = "2006-02-03"

gsci = financial_data(ticker, start_date, interval)
lambda_vals = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
model_gsci = regime_detection_tf(gsci, lambda_vals)
model_gsci.run_model(tuning=True)

macro_data_path = "current.csv"
model_type = 'logistic_regression'
gsci_prediction = regime_prediction_ml(model_gsci, macro_data_path, model_type=model_type, use_macro_data=True)

gsci_prediction.run()