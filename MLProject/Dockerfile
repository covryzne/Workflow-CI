FROM python:3.12.7-slim

WORKDIR /app

COPY MLProject/ .

RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy xgboost matplotlib seaborn python-dotenv

CMD ["mlflow", "run", ".", "--no-conda", "-P", "data_path=student_habits_preprocessing.csv"]