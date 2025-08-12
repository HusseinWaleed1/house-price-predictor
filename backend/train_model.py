import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load the dataset from sklearn
california_housing = fetch_california_housing(as_frame=True)
house_price_dataframe = california_housing.frame

# 2. Separate features and target
X = house_price_dataframe.drop(['MedHouseVal'], axis=1)
Y = house_price_dataframe['MedHouseVal']

# 3. Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 4. Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train the model (reduced size)
model = RandomForestRegressor(n_estimators=50, random_state=2)
model.fit(X_train, Y_train)

# 6. Evaluate the model
training_data_prediction = model.predict(X_train)
score_1 = metrics.r2_score(Y_train, training_data_prediction)
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
print("Training R squared error : ", score_1)
print('Training Mean Absolute Error : ', score_2)

test_data_prediction = model.predict(X_test)
score_3 = metrics.r2_score(Y_test, test_data_prediction)
score_4 = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("Test R squared error : ", score_3)
print('Test Mean Absolute Error : ', score_4)

# 7. Save the model and scaler (compressed)
joblib.dump(model, "models/house_model.pkl", compress=3)
joblib.dump(scaler, "models/scaler.pkl", compress=3)

print("✅ Model and scaler saved in models/")