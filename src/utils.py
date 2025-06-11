import joblib
import pandas as pd
import numpy as np
import json
import os
import traceback

# --- Configuration for file paths relative to the src directory ---
# utils.py is in src/, models/ is a sibling to src/ (i.e., in the project root)
# So, one level up from src/ to reach the project root, then into models/
MODELS_DIR_RELATIVE_TO_SRC = os.path.join(os.path.dirname(__file__), '..', 'models')

MODEL_FILE = os.path.join(MODELS_DIR_RELATIVE_TO_SRC, 'best_churn_model.pkl')
SCALER_FILE = os.path.join(MODELS_DIR_RELATIVE_TO_SRC, 'scaler.pkl')
ENCODER_COLUMNS_FILE = os.path.join(MODELS_DIR_RELATIVE_TO_SRC, 'encoder_columns.json')

# These lists MUST be consistent with notebook 01 (columns before OHE and scaling)
ORIGINAL_NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
ORIGINAL_CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen' # SeniorCitizen was treated as object/category
]

_model = None
_scaler = None
_encoder_columns = []
_models_loaded = False # Flag to track if assets are loaded

def load_prediction_assets():
    """Loads the model, scaler, and encoder columns from disk. Caches them for subsequent calls."""
    global _model, _scaler, _encoder_columns, _models_loaded
    
    if _models_loaded: # Return cached assets if already loaded
        return _model, _scaler, _encoder_columns

    try:
        # Check if models directory exists
        if not os.path.exists(MODELS_DIR_RELATIVE_TO_SRC):
             print(f"Error in src/utils.py: Models directory not found at {os.path.abspath(MODELS_DIR_RELATIVE_TO_SRC)}. Please run notebooks first to generate model files.")
             _models_loaded = False
             return None, None, [] # Return None for all if directory is missing

        _model = joblib.load(MODEL_FILE)
        _scaler = joblib.load(SCALER_FILE)
        with open(ENCODER_COLUMNS_FILE, 'r') as f:
            _encoder_columns = json.load(f)
        
        _models_loaded = True # Set flag to true only if all assets load successfully
        print("Prediction assets (model, scaler, encoder columns) loaded successfully from src/utils.py.")
        return _model, _scaler, _encoder_columns
    
    except FileNotFoundError as e:
        print(f"Error loading prediction assets in src/utils.py: {e}. One or more model files are missing. Ensure notebooks have been run successfully.")
        _models_loaded = False
        # Reset all to None if any file is missing to ensure consistent state
        _model, _scaler, _encoder_columns = None, None, [] 
        return None, None, []
    except Exception as e:
        print(f"An unexpected error occurred during asset loading in src/utils.py: {e}")
        traceback.print_exc()
        _models_loaded = False
        _model, _scaler, _encoder_columns = None, None, []
        return None, None, []

def preprocess_input_data(input_data_dict):
    """
    Preprocesses raw input data (from form) to match the model's expected format.
    Args:
        input_data_dict (dict): A dictionary where keys are feature names and values are user inputs.
    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction, or None if error.
    """
    # Ensure assets are loaded (or try to load them)
    model_asset, scaler_asset, encoder_columns_list_asset = load_prediction_assets()
    
    # Check if any crucial asset is missing after attempting to load
    if not model_asset or not scaler_asset or not encoder_columns_list_asset:
        print("Error in preprocess_input_data: Prediction assets not loaded. Cannot preprocess.")
        return None

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data_dict])

        # 1. Ensure SeniorCitizen is string (as it was treated as categorical for OHE in notebook)
        if 'SeniorCitizen' in input_df.columns:
            input_df['SeniorCitizen'] = input_df['SeniorCitizen'].astype(str)

        # 2. One-Hot Encode categorical features using ORIGINAL_CATEGORICAL_COLS
        # dummy_na=False ensures no columns are created for NaN values if any (though form should prevent them)
        input_df_encoded = pd.get_dummies(input_df, columns=ORIGINAL_CATEGORICAL_COLS, dummy_na=False)

        # 3. Align columns with training data (using _encoder_columns loaded from json)
        # Create a template DataFrame with all columns the model was trained on, initialized to 0
        aligned_df = pd.DataFrame(columns=encoder_columns_list_asset) 
        
        # Concatenate. This adds columns from input_df_encoded to aligned_df.
        # Columns in aligned_df not in input_df_encoded will be NaN, then filled with 0.
        # This step is crucial for handling cases where the input might not produce all OHE columns
        # (e.g., if a category seen during training is not in the current input).
        temp_df = pd.concat([aligned_df, input_df_encoded], ignore_index=False, sort=False)
        input_processed = temp_df.fillna(0)
        
        # Ensure only the encoder_columns_list_asset are present and in the correct order.
        # This handles cases where new, unseen categorical values in input might create extra columns.
        input_processed = input_processed[encoder_columns_list_asset]


        # 4. Scale numerical features (using ORIGINAL_NUMERICAL_COLS)
        # These are the original column names, not the OHE ones.
        cols_to_scale_in_input = [col for col in ORIGINAL_NUMERICAL_COLS if col in input_processed.columns]
        
        if cols_to_scale_in_input:
            input_processed[cols_to_scale_in_input] = scaler_asset.transform(input_processed[cols_to_scale_in_input])
        else:
            print("Warning in preprocess_input_data: No original numerical columns found to scale in the processed input. This might be an issue.")
            
        return input_processed

    except Exception as e:
        print(f"Error during input data preprocessing in src/utils.py: {e}")
        traceback.print_exc()
        return None

def make_prediction(processed_input_df):
    """
    Makes a churn prediction using the loaded model.
    Args:
        processed_input_df (pd.DataFrame): Preprocessed DataFrame.
    Returns:
        tuple (prediction_label (int), prediction_probability_churn (float)) or (None, None) if error.
    """
    model_asset, _, _ = load_prediction_assets() # Only model is needed here
    
    if not model_asset:
        print("Error in make_prediction: Model not loaded. Cannot predict.")
        return None, None
    if processed_input_df is None:
        print("Error in make_prediction: Processed input DataFrame is None. Cannot predict.")
        return None, None

    try:
        pred_proba_arr = model_asset.predict_proba(processed_input_df)
        prediction_proba_churn = pred_proba_arr[0][1]  # Probability of churn (class 1)
        
        churn_threshold = 0.5 # Example threshold
        prediction_label = 1 if prediction_proba_churn >= churn_threshold else 0
        
        return prediction_label, prediction_proba_churn
    except Exception as e:
        print(f"Error during prediction in src/utils.py: {e}")
        traceback.print_exc()
        return None, None

# Optional: Attempt to load assets when this module is first imported.
# This helps in identifying missing model files early when the Flask app starts.
# If loading fails, _models_loaded will remain False.
# load_prediction_assets()