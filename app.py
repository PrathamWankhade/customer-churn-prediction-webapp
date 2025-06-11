from flask import Flask, render_template, request
import os
import traceback

# Import helper functions and constants from src.utils
# Ensure src directory is treated as a package (e.g. by having an __init__.py in src/)
from src.utils import (
    preprocess_input_data, 
    make_prediction, 
    ORIGINAL_NUMERICAL_COLS, 
    ORIGINAL_CATEGORICAL_COLS,
    load_prediction_assets # To check status or trigger load
)

app = Flask(__name__)

# --- Attempt to load prediction assets on application startup ---
# This call will attempt to load the model, scaler, and encoder_columns.
# The _models_loaded flag within utils.py will be set.
# We can check its status for early warnings.
# The actual asset objects are kept within utils.py's scope.
initial_model, initial_scaler, initial_encoder_cols = load_prediction_assets()

if not initial_model or not initial_scaler or not initial_encoder_cols:
    print("WARNING (app.py): One or more prediction assets (model, scaler, encoder columns) could not be loaded on app startup. Predictions may fail. Please check logs from src/utils.py and ensure notebooks have been run successfully.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    # Preserve form data on POST for re-population, even if errors occur
    form_data = request.form.to_dict() if request.method == 'POST' else {}


    if request.method == 'POST':
        # Double-check if assets are truly available (in case of dynamic loading issues or first-time errors)
        # This re-calls load_prediction_assets which will return cached versions if already loaded,
        # or attempt to load again if failed initially.
        current_model, current_scaler, current_encoder_cols = load_prediction_assets()
        if not current_model or not current_scaler or not current_encoder_cols:
             prediction_text = "Critical Error: Model components are not available. Prediction service is offline. Please check server logs."
             return render_template('result.html', prediction_text=prediction_text, form_data=form_data)
        
        try:
            # Collect data from form
            input_data_dict = {}
            
            # Numerical features
            for col in ORIGINAL_NUMERICAL_COLS: # Use constants from utils
                form_val = request.form.get(col)
                # Basic validation for numerical inputs
                if form_val is None or form_val.strip() == '':
                    # Default to 0.0 or handle as an error if field is strictly required
                    input_data_dict[col] = 0.0 
                    print(f"Warning: Numerical field '{col}' was empty, defaulted to 0.0.")
                else:
                    input_data_dict[col] = float(form_val) # Can raise ValueError
            
            # Categorical features
            for col in ORIGINAL_CATEGORICAL_COLS: # Use constants from utils
                input_data_dict[col] = request.form.get(col, '') # Default to empty string if missing

            # Preprocess using utility function from src/utils.py
            processed_df = preprocess_input_data(input_data_dict)

            if processed_df is None:
                prediction_text = "Error during data preprocessing. Unable to make a prediction. Please check server logs."
            else:
                # Make prediction using utility function from src/utils.py
                prediction_label, prediction_proba_churn = make_prediction(processed_df)

                if prediction_label is None or prediction_proba_churn is None:
                    prediction_text = "Error during prediction. Unable to get a result. Please check server logs."
                else:
                    if prediction_label == 1:
                        prediction_text = f"LIKELY to Churn (Confidence: {prediction_proba_churn*100:.2f}%)"
                    else:
                        # Confidence for "Not Churn" is 1 - P(Churn)
                        prediction_text = f"UNLIKELY to Churn (Confidence: {(1-prediction_proba_churn)*100:.2f}%)"

        except ValueError as ve: # Specifically for float conversion errors
            prediction_text = f"Invalid input for a numerical field: {str(ve)}. Please ensure numbers are entered correctly."
            print(f"ValueError in app.py during form data collection: {ve}")
            traceback.print_exc()
        except Exception as e:
            prediction_text = f"An unexpected error occurred in the application: {str(e)}"
            print(f"Unexpected error in app.py main route: {e}")
            traceback.print_exc()

        return render_template('result.html', prediction_text=prediction_text, form_data=form_data)

    # For GET request, just render the empty form
    return render_template('index.html', form_data=form_data)


if __name__ == '__main__':
    # The load_prediction_assets() call at the top of the file already handles initial loading/checking.
    # No need for an explicit models directory check here if utils.py handles it.
    
    # Check if the 'src' directory (and thus utils.py) is accessible
    # This is more of a sanity check for the Python path
    try:
        from src import utils
        print("Successfully imported src.utils module.")
    except ImportError:
        print("ERROR: Could not import from 'src' directory. Ensure it's a package (contains __init__.py) and in PYTHONPATH if necessary.")
        # Potentially exit or raise error if src.utils is critical for app to even start
    
    app.run(debug=True, host='0.0.0.0', port=5000)