import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env from backend/.env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
)

MONGODB_DB_NAME = os.getenv(
    "MONGODB_DB_NAME",
    "automl_platform"
)

STORAGE_DIR = os.getenv(
    "STORAGE_DIR",
    "storage/datasets"
)

MAX_FILE_SIZE_MB = int(
    os.getenv("MAX_FILE_SIZE_MB", "50")
)

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx"
}


ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}