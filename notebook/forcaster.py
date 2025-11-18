import sys
import os
import tensorflow
print(sys.executable)

os.chdir(os.getcwd())
print("Current Working Directory after change:", os.getcwd())
import matplotlib.pyplot as plt

# Add the source directory to the system path
sys.path.append(os.path.abspath('../src'))
import utils
import importlib

import time
import warnings
import pandas as pd
import numpy as np
import math
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import pyodbc

def insert_update_on_db(df_in):
    print("aaaaaaaaaaaaaaaaaaaa")
    print(df_in)
    con_str = (
        # r'DRIVER={SQL Server Native Client 11.0};'
        r'DRIVER={SQL Server};'
        r'SERVER=ede-sqlserver16;'
        r'DATABASE=ANALYSIS;'
        r'UID=samueledelia;'
        r'PWD={Analysis_2025$}',
    )[0]

    df_in['FLOWQUARTER']=df_in['FLOWHOUR']
    df_in=df_in.reindex(columns=['FLOWDATE','FLOWHOUR','FLOWQUARTER','Predicted_SBIL_MWH','MACRO_INDEX','Run_Timestamp'])
 
    df_in.columns = ['FLOWDATE','FLOWHOUR','FLOWQUARTER','PREDICTED_SBIL_MWH','MACRO_INDEX','RUN_TIMESTAMP']
    df_in['RUN_TIMESTAMP'] = pd.to_datetime(df_in['RUN_TIMESTAMP'])
    con = pyodbc.connect(
        con_str,
        autocommit=False,
        readonly=False,
        fast_executemany=True
    )

    cur = con.cursor()
    

    cur.fast_executemany = True

    for index, row  in df_in.iterrows():
        cur.execute(
            fr'''
                IF EXISTS (SELECT 1 FROM  FORECAST_SIGN_QH WHERE FLOWDATE = ? AND FLOWHOUR = ? AND FLOWQUARTER = ? AND MACRO_INDEX = ?)
                    BEGIN
                    UPDATE FORECAST_SIGN_QH SET PREDICTED_SBIL_MWH = ?, RUN_TIMESTAMP = ? WHERE FLOWDATE = ? AND FLOWHOUR = ? AND FLOWQUARTER = ? AND MACRO_INDEX = ?;
                    END
                    ELSE
                    BEGIN
                    INSERT INTO FORECAST_SIGN_QH (FLOWDATE,FLOWHOUR,FLOWQUARTER,PREDICTED_SBIL_MWH,MACRO_INDEX,RUN_TIMESTAMP) VALUES (?, ?, ?, ?, ?, ?);
                    END
            ''',
            row['FLOWDATE'],
            row['FLOWHOUR'],
            row['FLOWQUARTER'],
            row['MACRO_INDEX'],
            row['PREDICTED_SBIL_MWH'],
            row['RUN_TIMESTAMP'],
            row['FLOWDATE'],
            row['FLOWHOUR'],
            row['FLOWQUARTER'],
            row['MACRO_INDEX'],
            row['FLOWDATE'],
            row['FLOWHOUR'],
            row['FLOWQUARTER'],
            row['PREDICTED_SBIL_MWH'],
            row['MACRO_INDEX'],
            row['RUN_TIMESTAMP'],

        ) 
            
    
    con.commit()
    cur.close()
    con.close()


# Reload the module
importlib.reload(utils)

# Load the prebuilt dataset from CSV instead of assembling it on the fly
df_nord_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'df_nord_h_port.csv'))
df_nord = pd.read_csv(df_nord_path, parse_dates=['ORAINI'])
df_nord = df_nord.set_index('ORAINI')

# Keep only the most recent observations used for training/prediction
df_nord = df_nord[df_nord.index >= '2024-08-27']

# Check for duplicate timestamps in the index and remove duplicates
df_nord = df_nord[~df_nord.index.duplicated(keep='first')]


# Check for duplicate timestamps in the index and remove duplicates
df_nord = df_nord[~df_nord.index.duplicated(keep='first')]

# Define past, future, and present covariates
past_covariates = []

future_covariates = [] #df_nord[['MGP_NORD_PURCHASES', 'MGP_NORD_SALES']]

present_covariates = df_nord[['SBIL_MWH_lag1', 'SBIL_MWH_lag2', 'SBIL_MWH_lag3', 'SBIL_MWH_lag24', 'UNBALANCE_IDRO-NON-PROGRAMMABILE_MACRONORD', 'UNBALANCE_IDRO-PROGRAMMABILE_NORD', 'UNBALANCE_SOLARE_NORD']]
target = df_nord['SBIL_MWH']

# Drop rows with NaN values resulting from the shift
df_nord = df_nord.dropna()

# Features (X) and Target (y)
X = df_nord.drop(columns=['SBIL_MWH'])
y = df_nord['SBIL_MWH']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Add the Dropout-enhanced MLP Model
model2 = Sequential()

# Input layer + First hidden layer with L2 regularization and Dropout
model2.add(Dense(8, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.2))  # Dropout with 20% probability

# Second hidden layer with L2 regularization and Dropout
model2.add(Dense(8, activation='relu', kernel_regularizer=l2(0.01)))
model2.add(Dropout(0.2))  # Dropout with 20% probability

# Output layer (single neuron for regression)
model2.add(Dense(1))

# Adam optimizer with initial learning rate 0.001
optimizer = Adam(learning_rate=0.001)

# Compile the model
model2.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model2.fit(X_train, y_train, epochs=150, batch_size=512, validation_split=0.2, callbacks=[early_stopping])



# Evaluate the model on test data
y_pred = model2.predict(X_test)

# Inverse transform predictions and true values to get the original scale
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Number of future steps to predict
n_future_steps = 4  # Predict 4 steps ahead

# Get the last timestamp from the dataset
last_timestamp = df_nord.index[-1]

# Get the most recent data from the test set for the starting point
last_input = X_test[-1].reshape(1, -1)  # Last row from test features

# Placeholder for predictions
predicted_values = []

# Monte Carlo Dropout
n_simulations = 100  # Perform 100 stochastic forward passes
mc_predictions = []  # Placeholder for stochastic predictions

current_input = last_input
for step in range(n_future_steps):
    # Collect multiple stochastic predictions for MC Dropout
    stochastic_results = []
    for _ in range(n_simulations):
        stochastic_pred = model2(current_input, training=True)  # Enable dropout during prediction
        stochastic_results.append(stochastic_pred.numpy().flatten()[0])
    stochastic_results = np.array(stochastic_results)

    # Calculate mean and standard deviation
    mean_prediction = stochastic_results.mean()
    std_prediction = stochastic_results.std()

    # Save the results
    predicted_values.append((mean_prediction, std_prediction))
    mc_predictions.append(stochastic_results)

    # Update the input for the next prediction
    current_input = np.roll(current_input, shift=-1, axis=1)  # Shift all features to the left
    current_input[0, -1] = mean_prediction  # Replace the last feature with the predicted value

# Extract means and standard deviations
predicted_means = [mean for mean, _ in predicted_values]
predicted_stds = [std for _, std in predicted_values]

# Convert to the original scale
predicted_means_orig = scaler_y.inverse_transform(np.array(predicted_means).reshape(-1, 1)).flatten()
predicted_stds_orig = scaler_y.inverse_transform(np.array(predicted_stds).reshape(-1, 1)).flatten()

# Calculate 95% confidence intervals
z_score = 1.96
ci_upper = predicted_means_orig + z_score * predicted_stds_orig
ci_lower = predicted_means_orig - z_score * predicted_stds_orig

# Add 1-hour intervals for the future steps
future_dates = [last_timestamp + pd.Timedelta(hours=i + 1) for i in range(n_future_steps)]

# Create a DataFrame to store the predicted future values
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_SBIL_MWH': predicted_means_orig,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper
})

# Debug: Confirm the DataFrame structure
print(f"Future DataFrame:\n{future_df}")

# Plotting
plt.figure(figsize=(12, 6))

# Get the last `n` actual observations (from the test set) for plotting
n_last_obs = 20  # For example, plot the last 20 observed values
last_observed_dates = df_nord.index[-n_last_obs:]  # Get the dates for the last observations
last_observed_values = scaler_y.inverse_transform(y_test[-n_last_obs:].reshape(-1, 1)).flatten()

last_observed_df = pd.DataFrame({
    'Date': last_observed_dates,
    'Actual_SBIL_MWH': last_observed_values
})

"""
# Plot observed values
plt.step(last_observed_df['Date'], last_observed_df['Actual_SBIL_MWH'], label='Observed SBIL_MWH', color='blue')


# Plot predicted values
plt.plot(future_df['Date'], future_df['Predicted_SBIL_MWH'], label='Predicted Future SBIL_MWH', color='red', linestyle='--', marker='o', markersize=10)

# Plot confidence intervals
plt.fill_between(
    future_df['Date'], future_df['CI_Lower'], future_df['CI_Upper'], color='red', alpha=0.2, label='95% Confidence Interval'
)

# Add labels and title
plt.title('Observed and Predicted Future SBIL_MWH Values with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('SBIL_MWH')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()

"""
# File path for storing predictions
file_path = r'C:\imbalance_forecast\data\forecast_segno_orario_1.csv'

# Step 1: Check if the file exists and is not empty
if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
    # Read the existing DataFrame from the CSV file
    df_existing = pd.read_csv(file_path)
else:
    # If the file does not exist or is empty, create an empty DataFrame with the necessary columns
    df_existing = pd.DataFrame(columns=['FLOWDATE', 'FLOWHOUR', 'Predicted_SBIL_MWH', 'Run_Timestamp', 'MACRO_INDEX'])

# Step 2: Assign predictions from `future_df` to respective variables
# Assuming `future_df` has predictions for h+1 in row 0, h+2 in row 1, etc.
predictions = [
    round(future_df.loc[0, 'Predicted_SBIL_MWH'], 3),
    round(future_df.loc[1, 'Predicted_SBIL_MWH'], 3),
    round(future_df.loc[2, 'Predicted_SBIL_MWH'], 3),
    round(future_df.loc[3, 'Predicted_SBIL_MWH'], 3)
]

# Use the timestamp of the h+1 prediction for the new rows
run_timestamp = datetime.now()
future_date = pd.to_datetime(future_df.loc[0, 'Date'])  # Ensure future_date is a datetime object
flow_date = int(future_date.strftime('%Y%m%d'))  # FLOWDATE in YYYYMMDD format
flow_hour = future_date.hour  # FLOWHOUR extracted as the hour
macro_index = 'MACRONORD'

# Step 3: Create individual rows for h+1 to h+4
new_rows = []
for i, predicted_value in enumerate(predictions):
    new_row = {
        'FLOWDATE': flow_date,
        'FLOWHOUR': flow_hour + i,  # Increment the hour for each prediction
        'Predicted_SBIL_MWH': float(predicted_value),
        'Run_Timestamp': run_timestamp,
        'MACRO_INDEX': macro_index
    }
    new_rows.append(new_row)

# Step 4: Convert the new rows into a DataFrame
new_rows_df = pd.DataFrame(new_rows)

# Step 5: Append the new rows to the existing DataFrame
df_existing = pd.concat([df_existing, new_rows_df], ignore_index=True)

# Step 6: Save the updated DataFrame back to the CSV file
df_existing.to_csv(file_path, index=False)
insert_update_on_db(new_rows_df)


print("Data successfully appended or updated in the file.")
