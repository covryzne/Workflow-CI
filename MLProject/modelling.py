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
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        
        mlflow.log_param("model_type", model_name)
        if model_name == "Random Forest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        elif model_name == "XGBoost":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("learning_rate", model.learning_rate)
        
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("explained_variance", explained_var)
        
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)
        
        plot_dir = "Membangun_model/Actual VS Predicted Graph"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model_name}_prediksi.png")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Predicted vs Actual ({model_name})')
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        
        print(f"{model_name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, Explained Variance: {explained_var:.4f}")

def main():
    # Untuk test lokal, uncomment baris ini dan comment DagsHub
    # mlflow.set_tracking_uri("http://localhost:5000")
    # Untuk DagsHub
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/covryzne/Eksperimen_SML_ShendiTeukuMaulanaEfendi.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'covryzne'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
    
    mlflow.set_experiment("Student_Performance_Prediction")
    df = pd.read_csv('student_habits_preprocessing.csv')
    
    X = df.drop('exam_score', axis=1)
    y = df['exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ("XGBoost", XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ]
    
    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()