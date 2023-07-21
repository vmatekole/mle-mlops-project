import os
from dotenv import load_dotenv

import pandas as pd

import mlflow
from mlflow.tracking.client import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from rich import print
from rich.console import Console

# year = 2021
# month = 9
# color = "green"

# # Download the data
# if not os.path.exists(f"./data/{color}_tripdata_{year}-{month:02d}.parquet"):
#     os.system(
#         f"wget -P ./data https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet")

# Load the data

df = pd.read_parquet(f"./data/training/dvc_train.parquet")

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Set up the connection to MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# Setup the MLflow experiment
mlflow.set_experiment("green-taxi-monitoring-project")

features = ["PULocationID", "DOLocationID", "trip_distance",
            "passenger_count", "fare_amount", "total_amount"]
target = 'duration'

# calculate the trip duration in minutes and drop trips that are less than 1 minute and more than 2 hours


def calculate_trip_duration_in_minutes(df):
    df["duration"] = (df["lpep_dropoff_datetime"] -
                      df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 8)]
    df = df[features + [target]]
    return df


def main():        
    df_processed = calculate_trip_duration_in_minutes(df)

    y = df_processed["duration"]
    X = df_processed.drop(columns=["duration"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2)

    IN_GOOGLE_CLOUD: str = os.getenv("IN_GOOGLE_CLOUD")    
    if not IN_GOOGLE_CLOUD:        
        SA_KEY = os.getenv("SA_KEY")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_KEY

    with mlflow.start_run():
        tags = {
            "model": "linear regression",
            "developer": "Victor Matekole",
            "dataset": f"Defined in DVC/Git — Should provide the hash and details from DVC/Git",            
            "features": features,
            "target": target
        }
        mlflow.set_tags(tags)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(lr, "model")
        run_id = mlflow.active_run().info.run_id

        model_uri = f"runs:/{run_id}/model"
        model_name = "green-taxi-ride-duration-project"
        mlflow.register_model(model_uri=model_uri, name=model_name)
        models = client.get_latest_versions(model_name, stages=["None"])
        model_version = models[0].version
        new_stage = "Production"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=new_stage,
            archive_existing_versions=True
        )


if __name__ == '__main__':
    main()
