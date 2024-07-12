import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the employee data
employee_data = np.array([
    [28, 50000, 4],
    [32, 60000, 7],
    [25, 45000, 2],
    [30, 55000, 5],
    [35, 70000, 9],
    [27, 48000, 3],
    [29, 52000, 6],
    [31, 58000, 8],
    [26, 46000, 1],
    [33, 65000, 10]
])

# Define feature names
feature_names = ['age', 'salary', 'experience']

# Create labels
labels = np.array([8,7,6,9,8,6,7,8,5,9])  # Assuming 'performance' as the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(employee_data, labels, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data (using the same scaler fitted on training data)
X_test_scaled = scaler.transform(X_test)

# Initialize RandomForestRegressor for predicting continuous "experience"
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the regressor to the training data
regressor.fit(X_train_scaled, y_train)

# Predict "experience" on the test data
y_pred = regressor.predict(X_test_scaled)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print feature importances based on RandomForestRegressor
print("Feature Importances:")
for feature, importance in zip(feature_names, regressor.feature_importances_):
    print(f"{feature}: {importance}")

# Example prediction for a new data point (if needed)
new_data_point = np.array([[32, 60000, 7]])  # Example new data point (age, salary, performance)
predicted_performance = regressor.predict(scaler.transform(new_data_point))
print(f"Predicted Performance: {predicted_performance}")