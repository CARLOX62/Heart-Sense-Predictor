from flask import Flask, render_template, request, flash, redirect, url_for, send_file
import numpy as np
import joblib
import os
import pandas as pd
from io import BytesIO # For in-memory file creation

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here' # Required for flash messages

# --- Configuration ---
MODEL_PATH = "best_heart_failure_model.pkl"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "heart_failure_clinical_records_dataset (1).csv"

# --- Global Variables for Model and Scaler ---
model = None
scaler = None
feature_names = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]

# --- Load Model and Scaler on App Startup ---
def load_resources():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Error: Model file not found at {MODEL_PATH}. Prediction will not work.")
            flash(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the correct directory.", 'error')

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded successfully from {SCALER_PATH}")
        else:
            print(f"Error: Scaler file not found at {SCALER_PATH}. Prediction will not work correctly.")
            flash(f"Error: Scaler file '{SCALER_PATH}' not found. Please ensure it's in the correct directory.", 'error')

    except Exception as e:
        print(f"An error occurred during resource loading: {e}")
        flash(f"An error occurred during application startup: {e}", 'error')

# Call resource loading function when the app starts
with app.app_context():
    load_resources()

# --- Helper Function for Input Validation ---
def validate_and_parse_features(form_data):
    parsed_features = {}
    errors = []

    feature_specs = {
        'age': {'type': float, 'min': 0, 'max': 120, 'placeholder': 'e.g., 60'},
        'anaemia': {'type': int, 'options': [0, 1], 'placeholder': '0 or 1'},
        'creatinine_phosphokinase': {'type': float, 'min': 0, 'placeholder': 'e.g., 582'},
        'diabetes': {'type': int, 'options': [0, 1], 'placeholder': '0 or 1'},
        'ejection_fraction': {'type': float, 'min': 0, 'max': 100, 'placeholder': 'e.g., 38'},
        'high_blood_pressure': {'type': int, 'options': [0, 1], 'placeholder': '0 or 1'},
        'platelets': {'type': float, 'min': 0, 'placeholder': 'e.g., 263358.03'},
        'serum_creatinine': {'type': float, 'min': 0, 'placeholder': 'e.g., 1.1'},
        'serum_sodium': {'type': float, 'min': 0, 'placeholder': 'e.g., 136'},
        'sex': {'type': int, 'options': [0, 1], 'placeholder': '0 or 1'},
        'smoking': {'type': int, 'options': [0, 1], 'placeholder': '0 or 1'},
        'time': {'type': float, 'min': 0, 'placeholder': 'e.g., 6'},
    }

    for feature_name in feature_names:
        value_str = form_data.get(feature_name)
        spec = feature_specs.get(feature_name, {})

        if not value_str:
            errors.append(f"'{feature_name}' is required.")
            continue

        try:
            if spec.get('type') == int:
                value = int(float(value_str))
            else:
                value = float(value_str)
            parsed_features[feature_name] = value

            if 'min' in spec and value < spec['min']:
                errors.append(f"'{feature_name}' must be at least {spec['min']}.")
            if 'max' in spec and value > spec['max']:
                errors.append(f"'{feature_name}' must be at most {spec['max']}.")
            if 'options' in spec and value not in spec['options']:
                errors.append(f"'{feature_name}' must be one of {spec['options']}.")

        except ValueError:
            errors.append(f"'{feature_name}' must be a valid number.")
        except Exception as e:
            errors.append(f"An unexpected error occurred with '{feature_name}': {e}")

    return parsed_features, errors

# --- Routes ---
@app.route('/')
def home():
    sample_data_html = ""
    try:
        if os.path.exists(DATASET_PATH):
            df_sample = pd.read_csv(DATASET_PATH)
            sample_data_html = df_sample.head().to_html(classes='data-table', index=False)
        else:
            sample_data_html = "<p>Sample dataset not found. Please ensure 'heart_failure_clinical_records_dataset (1).csv' is available.</p>"
    except Exception as e:
        sample_data_html = f"<p>Error loading sample data: {e}</p>"

    return render_template('index.html', sample_data_html=sample_data_html)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if model is None or scaler is None:
            flash("Prediction service is not available. Model or scaler failed to load.", 'error')
            return redirect(url_for('home'))

        parsed_features, errors = validate_and_parse_features(request.form)

        if errors:
            for error in errors:
                flash(error, 'error')
            # Pass form_data back to re-populate fields and prediction_result as None
            return render_template('index.html', form_data=request.form, prediction_result=None)

        features_list = [parsed_features[key] for key in feature_names]

        try:
            input_array = np.array([features_list])
            input_scaled = scaler.transform(input_array)

            prediction_label = model.predict(input_scaled)[0]
            prediction_proba = None

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                prediction_proba = proba[1] if prediction_label == 1 else proba[0]

            result_message = ""
            if prediction_label == 1:
                result_message = "⚠️ High Risk of Death"
                if prediction_proba is not None:
                    result_message += f" (Probability: {prediction_proba:.2%})"
                flash(result_message, 'warning')
            else:
                result_message = "✅ Low Risk of Death"
                if prediction_proba is not None:
                    result_message += f" (Probability: {prediction_proba:.2%})"
                flash(result_message, 'success')

            # Store prediction details in session or pass directly for download
            # For simplicity, we'll pass it directly to the template and use a hidden field for download
            # A more robust solution for download would involve storing in session or a temporary file
            # and generating a unique ID for download.
            
            # Prepare data for the prediction result table
            prediction_details = {
                "input_features": parsed_features,
                "prediction_label": prediction_label,
                "prediction_message": result_message,
                "prediction_probability": prediction_proba
            }

            return render_template('index.html',
                                   form_data=request.form,
                                   prediction_result=result_message,
                                   prediction_details=prediction_details) # Pass details for download

        except Exception as e:
            flash(f"An unexpected error occurred during prediction: {str(e)}", 'error')
            print(f"Prediction error: {e}")
            return render_template('index.html', form_data=request.form, prediction_result=None)

@app.route('/download_prediction', methods=['POST'])
def download_prediction():
    # Retrieve prediction details from hidden fields in the form
    # This is a simple way; for more complex data, consider session or database
    input_data_str = request.form.get('download_input_data')
    prediction_message = request.form.get('download_prediction_message')
    prediction_proba_str = request.form.get('download_prediction_probability')

    if not input_data_str or not prediction_message:
        flash("No prediction data available for download.", 'error')
        return redirect(url_for('home'))

    # Reconstruct input data from string
    input_data = {}
    try:
        # Assuming input_data_str is a JSON string of the parsed_features
        import json
        input_data = json.loads(input_data_str)
    except json.JSONDecodeError:
        flash("Error decoding input data for download.", 'error')
        return redirect(url_for('home'))

    output = BytesIO()
    # Create a simple text file for download
    output.write(b"--- Heart Failure Prediction Report ---\n\n")
    output.write(f"Prediction: {prediction_message}\n".encode('utf-8'))
    if prediction_proba_str:
        output.write(f"Probability: {float(prediction_proba_str):.2%}\n".encode('utf-8'))
    output.write(b"\n--- Input Features ---\n")
    for key, value in input_data.items():
        output.write(f"{key.replace('_', ' ').title()}: {value}\n".encode('utf-8'))

    output.seek(0) # Go to the beginning of the BytesIO object

    return send_file(output,
                     mimetype='text/plain',
                     as_attachment=True,
                     download_name='heart_failure_prediction_report.txt')


if __name__ == '__main__':
    app.run(debug=True)

