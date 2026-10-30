"""
Configuration file for Disaster Prediction & Response System
"""
import os

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
KAGGLE_DIR = os.path.join(BASE_DIR, '.kaggle')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(KAGGLE_DIR, exist_ok=True)

# Kaggle dataset identifiers
KAGGLE_DATASETS = {
    'flood': [
        'alanascharya/weather-related-disasters-from-1900-2021',
        'nathanlauga/historic-floods',
        'cityofLA/la-flood-zones',
        'kkhandekar/all-datasets-from-em-dat',  # Contains flood disasters
        'shailendramohite/natural-disasters-data',  # Multiple disaster types
        'utkarshx/gdelt-data-on-indian-floods',  # Indian floods
        'dataenthusiast123/natural-disaster-datasets'  # General disasters
    ],
    'earthquake': [
        'usgs/earthquake-database',
        'alessandrolobello/the-ultimate-earthquake-dataset',
        'warcoder/earthquake-dataset',
        'danielpe/usgs-earthquake-data'
    ]
}

# Model configurations
ML_MODELS = {
    'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
    'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': 42},
    'lstm': {'units': 50, 'epochs': 50, 'batch_size': 32}
}

# Gemini AI configuration
GEMINI_API_KEY = None  # Will be set from user input

# App settings
APP_TITLE = "AI-Based Disaster Prediction & Response System"
APP_ICON = "🌍"
