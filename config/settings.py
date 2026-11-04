import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Key Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# Application Credentials
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
APP_ENV = os.getenv("APP_ENV", "development")

# Image Preprocessing Policy
# Options: 'letterbox' (aspect-preserving padding with neutral gray borders) or 'baseline' (anamorphic 224x224 resize)
DEFAULT_PREPROCESSING_MODE = os.getenv("DEFAULT_PREPROCESSING_MODE", "letterbox")

# Ensure required directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
