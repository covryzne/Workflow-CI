import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score
import os
from dotenv import load_dotenv
load_dotenv()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        # Debug: Cek MLflow tracking URI
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Hitung metrik
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        
        # Log parameter
        mlflow.log_param("model_type", model_name)
        if model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        elif model_name == "XGBoost":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
        
        # Log metrik
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("explained_variance", explained_var)
        
        # Log model
        try:
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, model_name, input_example=X_test[:1])
                print(f"Logged XGBoost model to artifacts/{model_name}")
            else:
                mlflow.sklearn.log_model(model, model_name, input_example=X_test[:1])
                print(f"Logged sklearn model to artifacts/{model_name}")
        except Exception as e:
            print(f"Failed to log model {model_name}: {str(e)}")
            raise
        
        # Buat dan simpen plot
        plot_dir = "Membangun_model/Actual VS Predicted Graph"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model_name}_prediksi.png")
        
        try:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Predicted vs Actual ({model_name})')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")
        except Exception as e:
            print(f"Failed to save plot {plot_path}: {str(e)}")
            raise
        
        # Log plot sebagai artifact
        try:
            mlflow.log_artifact(plot_path, artifact_path=f"plots/{model_name}")
            print(f"Logged artifact {plot_path} to artifacts/plots/{model_name}")
        except Exception as e:
            print(f"Failed to log artifact {plot_path}: {str(e)}")
            raise
        
        # Debug
        print(f"MLflow artifact root: {os.getenv('MLFLOW_ARTIFACT_ROOT', 'mlruns')}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Plot path exists: {os.path.exists(plot_path)}")
        print(f"Run ID: {run.info.run_id}")
        
        # Debug: Cek artifacts di DagsHub
        client = mlflow.tracking.MlflowClient()
        try:
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"Artifacts for run_id {run.info.run_id}: {[a.path for a in artifacts]}")
        except Exception as e:
            print(f"Failed to list artifacts for run_id {run.info.run_id}: {str(e)}")
        
        # Cetak run_id untuk GitHub Actions
        run_id = run.info.run_id
        print(f"MLFLOW_RUN_ID={run_id}")
        
        print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, Explained Variance: {explained_var:.4f}")

def main():
    # Validasi DagsHub token
    if not os.getenv('DAGSHUB_TOKEN'):
        raise ValueError("DAGSHUB_TOKEN not set in environment")
    
    # Set DagsHub tracking
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/covryzne/Eksperimen_SML_ShendiTeukuMaulanaEfendi.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'covryzne'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
    
    mlflow.set_experiment("Student_Performance_Prediction")
    
    # Baca dataset
    try:
        df = pd.read_csv('student_habits_preprocessing.csv')
    except Exception as e:
        print(f"Failed to read dataset: {str(e)}")
        raise
    
    X = df.drop('exam_score', axis=1)
    y = df['exam_score']
    print(f"Feature columns: {X.columns.tolist()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    
    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()