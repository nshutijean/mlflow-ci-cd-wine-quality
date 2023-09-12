import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Fetching data from an online source
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(data_url, sep=";")

# print(data.columns)

# save the data to csv
# data.to_csv("winequality-red.csv", index=False)

# Splitting the data into training and testing sets
train, test = train_test_split(data, test_size=0.2)

# Defining the model
model = RandomForestRegressor(n_estimators=100, max_depth=10)

# Training the model
model.fit(train.drop(["quality"], axis=1), train["quality"])

# Logging the necessary data with MLflow
with mlflow.start_run():

    # Logging parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    
    predictions = model.predict(test.drop(["quality"], axis=1))

    # Logging the model
    mlflow.sklearn.log_model(model,"randomforestregressor")

    # Defining metrics
    mse = mean_squared_error(test["quality"], predictions)
    mae = mean_absolute_error(test["quality"], predictions)
    r2 = r2_score(test["quality"], predictions)

    # Logging metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Logging artifacts
    mlflow.log_artifact("winequality-red.csv")

    print("R-squared:", r2)

