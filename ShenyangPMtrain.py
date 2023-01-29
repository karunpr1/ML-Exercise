import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("ShenyangPMtrain.csv")

# Preprocessing
# Handle missing values
df = df.fillna(df.mean())

# Encode categorical variables
le = LabelEncoder()
df['cbwd'] = le.fit_transform(df['cbwd'])

# Split the data into features and target
X = df.drop(['PM_Taiyuanjie'], axis=1)
y = df['PM_Taiyuanjie']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Train the model
model = RandomForestRegressor(random_state=0)

# Create a grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_

# Train a final model using the best hyperparameters
final_model = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'], random_state=0)
final_model.fit(X_train, y_train)

# Evaluate the final model on the testing data
y_pred = final_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)