FROM python:3.12-slim

RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY MLProject/ .

RUN pip install --no-cache-dir \
    mlflow==2.17.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==2.1.2 \
    xgboost==2.1.1 \
    matplotlib==3.9.2 \
    seaborn==0.13.2 \
    python-dotenv==1.0.1

CMD ["mlflow", "run", ".", "--no-conda", "-P", "data_path=student_habits_preprocessing.csv"]