from flask import Flask, request, jsonify
import os
import joblib
import json
import traceback
from datetime import datetime
from uuid import uuid4

from modules.models import train_model # Import model training and dataset builder functions.
from modules.data import dataset_builder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Resolve base directories relative to this script for consistency.
MODEL_DIR = os.path.join(BASE_DIR, "storage", "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

app = Flask(__name__)

def load_latest_file(directory, extension=".joblib"):
    """
    Load the latest file with a given extension from a directory.
    Assumes files are named with sortable timestamps for lexical max.
    
    Parameters:
        directory (str): Path to directory to search.
        extension (str): File extension filter (default '.joblib').
        
    Returns:
        str or None: Path to the latest file or None if none found.
    """
    if not os.path.exists(directory):

        return None
    
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    
    if not files:

        return None
    
    latest_file = max(files) # Relies on timestamped filenames sorting lexically.

    return os.path.join(directory, latest_file)

def load_model_and_preprocessor():
    """
    Load the most recent trained model and preprocessor from storage.
    
    Returns:
        model: The trained model loaded via joblib or None if not found.
        preprocessor: The preprocessing pipeline loaded via joblib or None if not found.
        model_path: Path to the model file loaded or None.
        preprocessor_path: Path to the preprocessor file loaded or None.
    """
    model_path = load_latest_file(MODEL_DIR, ".joblib")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.joblib")

    model = None
    preprocessor = None

    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
    else:
        print("No trained model found.")

    if os.path.exists(preprocessor_path):
        print(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
    else:
        print("No preprocessor found.")

    return model, preprocessor, model_path, preprocessor_path

model, preprocessor, MODEL_PATH, PREPROCESSOR_PATH = load_model_and_preprocessor() # Load model and preprocessor once on startup.

def log_request(request_data, response_data):
    """
    Log incoming API request and corresponding response as a JSON file.
    Logs are stored with a timestamp and unique ID to support audit trails.
    
    Parameters:
        request_data (dict): The incoming request JSON data.
        response_data (dict): The API response JSON data.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = uuid4().hex[:6]
    log_path = os.path.join(LOG_DIR, f"log_{timestamp}_{unique_id}.json")

    with open(log_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'request': request_data,
            'response': response_data
        }, f, indent=2)

@app.route('/predict/', methods=['POST'])
def predict():
    """
    API endpoint to predict fraud status for a given transaction.
    Expects a JSON payload with required transaction fields.
    
    Returns:
        JSON with 'predict': 'fraudulent' or 'legitimate' on success,
        or error message with appropriate HTTP status code.
    """
    global model, preprocessor

    if model is None or preprocessor is None:

        return jsonify({'error': 'Model and preprocessor not loaded. Train the model first.'}), 500

    try:
        input_data = request.get_json()
        if not input_data:

            return jsonify({'error': 'No input data provided'}), 400

        required_fields = [
            'trans_date_trans_time', 'cc_num', 'unix_time', 'merchant',
            'category', 'amt', 'merch_lat', 'merch_long'
        ] # Check for required fields in input JSON.

        for field in required_fields:
            if field not in input_data:

                return jsonify({'error': f'Missing field: {field}'}), 400

        df = dataset_builder.process_input_for_prediction(input_data) # Convert input JSON to DataFrame suitable for prediction pipeline.

        X_processed = preprocessor.transform(df) # Apply preprocessing transforms.

        prediction_binary = model.predict(X_processed)[0] # Predict fraud class (0 or 1).
        prediction_label = 'fraudulent' if prediction_binary == 1 else 'legitimate'
        result = {'predict': prediction_label}

        log_request(input_data, result) # Log the request and response.

        return jsonify(result)

    except Exception as e:
        traceback.print_exc() # Print traceback for debugging and return error message.

        return jsonify({'error': str(e)}), 500

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    """
    API endpoint to trigger dataset creation from raw data sources.
    
    Returns:
        JSON with confirmation message and dataset file path on success,
        or error message with HTTP 500 on failure.
    """
    try:
        output_path = dataset_builder.build_dataset()
        response = {'message': 'Dataset created successfully.', 'path': output_path}
        log_request({}, response)

        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()

        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train():
    """
    API endpoint to trigger model training on the latest dataset.
    
    Returns:
        JSON with confirmation message and training metrics on success,
        or error message with HTTP 500 on failure.
    """
    global model, preprocessor, MODEL_PATH, PREPROCESSOR_PATH

    try:
        
        metrics = train_model.train_and_save_model() # Train model and save to disk.
        model, preprocessor, MODEL_PATH, PREPROCESSOR_PATH = load_model_and_preprocessor() # Reload the newly trained model and preprocessor.
        response = {'message': 'Model trained and saved.', 'training_metrics': metrics}

        log_request({}, response)

        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()

        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) # Run Flask server on all interfaces with debug enabled.