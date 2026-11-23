# ***SecureBank* Fraud Detection System**

This project provides a lightweight, containerized fraud detection system designed for the *SecureBank* Case Study and accompanying assignment(s). It includes data handling, model training, evaluation, a predictive API using Flask, and Docker-based deployment.

## **Project Structure**

```
securebank/
│
├── data_sources/                  # Raw datasets provided externally (excluded from repo).
│   ├── customer_release.csv
│   ├── fraud_release.json
│   └── transactions_release.parquet
│
├── executables/                   # Bash scripts for automated testing and management.
│   ├── create_dataset.sh          # Script to test dataset creation endpoint.
│   ├── predict.sh                 # Script to simulate fraud prediction requests.
│   ├── run_server.sh              # Script to launch the Flask API server.
│   └── train_model.sh             # Script to test model training endpoint.
│
├── modules/
│   ├── data/
│   │   ├── dataset_builder.py     # Generate engineered datasets from raw data sources.
│   │   └── raw_data_handler.py    # Class for loading and processing raw data files.
│   └── models/
│       ├── evaluate_results.py    # Evaluate trained models and generate reports/plots.
│       └── train_model.py         # Train and save machine learning models with versioning.
│
├── storage/
│   ├── datasets/                  # Generated datasets (train/test/full) with timestamps.
│   ├── models/                    # Saved models, preprocessors, and training metrics.
│   └── results/                   # Evaluation plots and JSON metrics.
│
├── .dockerignore                  # Docker ignore rules.
├── .gitignore                     # Git ignore rules.
├── app.py                         # Flask app serving inference and training endpoints.
├── Dockerfile                     # Docker container specification.
├── README.md                      # This documentation file.
├── requirements.txt               # Python dependencies for the system.
├── simple_tutorial.py             # (Optional) Tutorial or example usage script.
├── test_datahandler.py            # Unit tests for raw data handling modules.
├── test.json                      # Sample transaction JSON for inference testing.
└── test.py                        # (Optional) Additional test scripts.
```

## **Brief Overview**

* **data\_sources/**: Contains the original raw datasets provided for the assignment (excluded from the repository). You must place the provided CSV, JSON, and Parquet files here during evaluation.

* **executables/**: Contains bash scripts to automate testing and interaction with the SecureBank API. These scripts cover key workflows:

  * `create_dataset.sh`: Triggers dataset creation from raw data sources.
  * `train_model.sh`: Triggers training of the fraud detection model.
  * `predict.sh`: Sends transaction JSON data to the API for fraud prediction.
  * `run_server.sh`: Builds and runs the Docker containerized API server.

* **modules/data/dataset\_builder.py**: Loads raw data and performs comprehensive feature engineering, including time-based features (transaction month, day, weekday), fraud rate encoding, geographic distance calculations, and customer-wise train/test splitting to avoid data leakage.

* **modules/data/raw\_data\_handler.py**: Provides the `RawDataHandler` class responsible for loading, merging, cleaning, and summarizing raw data sources, facilitating dataset creation and exploration.

* **modules/models/train\_model.py**: Loads engineered datasets, preprocesses features with cardinality-aware encoders, trains a Random Forest classifier, evaluates performance metrics (precision, recall, F1), and saves trained models and preprocessors with timestamped versioning and “latest” pointers for reproducibility.

* **modules/models/evaluate\_results.py**: Loads the latest or specified trained model and dataset, preprocesses features, generates predictions, computes evaluation metrics (classification report, confusion matrix, ROC and precision-recall curves), and saves numerical metrics (JSON) and plots (PNG) for audit and monitoring.

* **modules/models/model\_dummy.py**: A placeholder model implementation that randomly returns fraud predictions, used during initial development stages before integrating the real model pipeline.

* **app.py**: The Flask web application exposing REST endpoints:

  * `/predict/` for real-time fraud prediction on transaction data.
  * `/create_dataset` to trigger dataset generation from raw sources.
  * `/train_model` to trigger model training.
    Each endpoint includes comprehensive logging of requests and responses, saved as JSON files with timestamps and unique IDs in `securebank/logs/`, enabling audit trails and system monitoring.

* **storage/**: Organized directory for persistent storage of generated datasets, trained models and preprocessors, evaluation results, plots, and training metrics. All files use timestamped naming conventions to ensure traceability and reproducibility.

* **requirements.txt**: Enumerates all required Python packages and dependencies needed to run the system both locally and within the Docker container.

* **test\_datahandler.py**: Unit test suite validating data loading, transformation, and summary statistics provided by the `RawDataHandler`.

* **test.json**: Sample JSON transaction record used to test the `/predict/` endpoint manually or via scripts.

## **Local Setup**

Ensure you have Python 3.9 or later installed, then set up a virtual environment and install all necessary dependencies:

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## **Running the System Locally**

First, build the dataset. Run the following command from the root directory:

```
python securebank/modules/data/dataset_builder.py
```

This will load raw data from `data_sources/`, engineer features, and save train/test/full datasets under `storage/datasets/`.

Next, train the model. Run the following command from the root directory:

```
python securebank/modules/models/train_model.py
```

This trains a Random Forest model on the latest dataset and saves the trained model and preprocessor under `storage/models/`.

Lastly, evaluate the model. Run the following command from the root directory:

```
python securebank/modules/models/evaluate_results.py
```

This will run an evaluation on the latest model and dataset, producing classification reports, confusion matrices, ROC, and precision-recall plots saved under `storage/results/`.

## **Running the System via Flask API**

First, start the Flask API server. Run the following command from the root directory:

```
securebank/executables/run_server.sh
```

If not already logged into your local Docker Desktop application, be sure to run the following command from the terminal:

```
docker logout
docker login # Logging out and back in will refresh Docker typically.
docker pull python:3.9-slim # Ensuring that the correct Python version is used just in case.
```

Running the first executable command launches the Flask server inside a Docker container, which exposes the `/predict/`, `/create_dataset`, and `/train_model` endpoints. The server listens on all interfaces (`0.0.0.0`) by default, typically on port 5000.

Next, create the dataset from raw data sources. Open a new terminal window and run:

```
securebank/executables/create_dataset.sh
```

This sends a POST request to the API’s `/create_dataset` endpoint, triggering feature engineering and dataset generation. The generated dataset is saved under `storage/datasets/`.

Next, train the model. In the same or a new terminal window, run:

```
securebank/executables/train_model.sh
```

This sends a POST request to `/train_model`, starting model training on the latest dataset. The trained model and preprocessor are saved with versioning under `storage/models/`.

Lastly, make a fraud prediction request using a sample transaction (such as `test.json`). Run:

```
securebank/executables/predict.sh
```

This sends a POST request to `/predict/` with the sample transaction data and returns whether it’s predicted to be `fraudulent` or `legitimate`.

This sequence mimics a full production workflow: data preparation, model training, and inference via the Flask API, all testable via simple bash scripts for convenience.