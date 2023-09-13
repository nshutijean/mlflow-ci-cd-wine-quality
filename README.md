## Wine quality MLflow pipeline

This project aims to predict the quality of wine based on quantitative features like the wine’s “fixed acidity”, “pH”, “residual sugar”, and so on, tracking and managing experiments using MLFlow. We will use the Wine Quality dataset from UCI’s machine learning repository. 

## Dependencies

- mlflow
- pandas
- sklearn


## Installation
To install the dependencies, you can use either pip or conda. Here are the steps:

Using pip: <br>
```
python -m venv mlflow-examples

source mlflow-examples/bin/activate

pip install -r requirements.txt
```

Using conda: <br>
```
conda install --yes --file requirements.txt
```

## Usage
To run the project, follow these steps:

- Clone this repository to your local machine.
- Navigate to the root directory of the project.
- Run `mlflow run .` to train the model and generate an artifact.
- Run `mlflow ui` to launch the MLFlow UI.
- Open your web browser and navigate to http://localhost:5000.
- In the MLFlow UI, you can view the experiment runs, metrics, and artifacts.

## Screenshots
