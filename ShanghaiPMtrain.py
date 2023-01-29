import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv("ShanghaiPMtrain.csv")

# Preprocessing
# Handle missing values
df = df.fillna(df.mean())

# Encode categorical variables
le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])

# Split the data into training and testing sets
X = df.drop(["PM_Jingan"], axis=1)
y = df["PM_Jingan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Create the random forest regressor object
rf = RandomForestRegressor()

# Use GridSearchCV to perform hyperparameters tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best set of hyperparameters
best_params = grid_search.best_params_

# Train the model
model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'],random_state=0)
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
