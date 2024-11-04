import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Load the CSV file
file_path = 'traffic.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Feature engineering: extract day of the week and hour from DateTime
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Day'] = data['DateTime'].dt.dayofweek  # Extract day of the week (0 = Monday, 6 = Sunday)
data['Hour'] = data['DateTime'].dt.hour  # Extract hour of the day

# Selecting features (Day, Hour, Junction) and target (Vehicles)
X = data[['Day', 'Hour', 'Junction']]
y = data['Vehicles']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: One-hot encode the 'Junction' feature
preprocessor = ColumnTransformer(transformers=[('junction', OneHotEncoder(), ['Junction'])], remainder='passthrough')

# Build the Random Forest regression pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 1: Function to predict traffic based on day, hour, and junction input
def predict_vehicle_count(day, hour, junction):
    input_data = pd.DataFrame([[day, hour, junction]], columns=['Day', 'Hour', 'Junction'])
    predicted_vehicles = model.predict(input_data)[0]
    return predicted_vehicles

# Example usage:
day_input = int(input("Enter the day of the week (0=Monday, 6=Sunday): "))
hour_input = int(input("Enter the hour of the day (0-23): "))
junction_input = int(input("Enter the junction number (1, 2, 3, or 4): "))

# Predict vehicle count for the given input
predicted_traffic = predict_vehicle_count(day_input, hour_input, junction_input)
print(f'Predicted vehicle count at Junction {junction_input} on day {day_input} at hour {hour_input}: {math.floor(predicted_traffic)}')

# Step 2: Create a heatmap of the predicted values for a particular junction

# Predict values across different days and hours for a specific junction
junction = 1  # Choose the junction for which to visualize the heatmap
day_range = np.arange(0, 7)  # Days of the week (0 = Monday, 6 = Sunday)
hour_range = np.arange(0, 24)  # Hours of the day

# Generate predictions for each combination of day and hour for the chosen junction
heatmap_data = np.zeros((len(day_range), len(hour_range)))

for i, day in enumerate(day_range):
    for j, hour in enumerate(hour_range):
        # Prepare input for prediction
        input_data = pd.DataFrame([[day, hour, junction]], columns=['Day', 'Hour', 'Junction'])
        heatmap_data[i, j] = model.predict(input_data)[0]

# Plot the heatmap using seaborn
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.1f', xticklabels=hour_range, yticklabels=day_range)

# Add labels and title
plt.title(f'Heatmap of Predicted Vehicle Count for Junction {junction}', fontsize=16)
plt.xlabel('Hour of the Day', fontsize=12)
plt.ylabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=12)

# Show the heatmap
plt.show()
