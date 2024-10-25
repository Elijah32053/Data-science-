import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Loading the dataset
accident_data = pd.read_csv('accident_data.csv')

# Dropping timestamp
accident_data.drop(columns=['timestamp'], inplace=True)

# Handle categorical variables using one-hot encoding
accident_data = pd.get_dummies(accident_data, drop_first=True)

# Separating features and target variable
X = accident_data.drop(columns=['accident_severity'])
y = accident_data['accident_severity']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Saving the model and scaler for future use
joblib.dump(model, 'accident_severity_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")