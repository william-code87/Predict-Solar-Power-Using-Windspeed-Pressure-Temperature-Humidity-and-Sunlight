import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
import glob
import joblib
from joblib import parallel_backend
import tempfile

# Load data
file_path_pattern = r'D:\PowerStation\Dataset\L*_Train.csv'
csv_files = glob.glob(file_path_pattern)
dataframes = [pd.read_csv(file) for file in csv_files]
data = pd.concat(dataframes, ignore_index=True)

# Feature Engineering: Create new features
data['Hour'] = pd.to_datetime(data['DateTime']).dt.hour
data['Day'] = pd.to_datetime(data['DateTime']).dt.day
data['Month'] = pd.to_datetime(data['DateTime']).dt.month

# List of numeric features for scaling
numeric_features = ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']

# Scale numeric features
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])
# Save the scaler for later use
scaler_path = r'D:\PowerStation\scalerMAE11.pkl'
joblib.dump(scaler, scaler_path)

# Define features and target
X = data.drop(['Power(mW)', 'DateTime'], axis=1)  # Keep all features except 'Power(mW)' and 'DateTime'
y = data['Power(mW)']  # Define target variable

# Apply One-Hot Encoding for 'LocationCode'
X = pd.get_dummies(X, columns=['LocationCode'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_jobs=2, random_state=42)

# Define the parameter search space for BayesSearchCV
param_space = {
    'n_estimators': (500, 1500),
    'learning_rate': (0.01, 0.3, 'log-uniform'),
    'max_depth': (3, 10),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'gamma': (0, 10),
    'reg_alpha': (0, 10),
    'reg_lambda': (0, 10)
}

# Setup Bayesian optimization search
opt = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_space,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_iter=500,  # Increase iterations
    n_jobs=2,  # Reduce parallel jobs
    random_state=42,
    verbose=2
)

# Set up temporary folder for joblib caching in D drive
temp_folder = tempfile.mkdtemp(dir='D:/')

# Run BayesSearchCV with parallel_backend and custom temp folder
with parallel_backend('loky', temp_folder=temp_folder):
    opt.fit(X_train, y_train)

# Best parameters from optimization
print(f"Best parameters: {opt.best_params_}")

# Evaluate the optimized model
y_pred = opt.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the best XGBoost model
joblib.dump(opt.best_estimator_, 'xgboost_optimized_model.pkl')
