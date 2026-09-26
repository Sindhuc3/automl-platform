from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

from app.config import MONGODB_URI, MONGODB_DB_NAME


client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=3000
)

database = client[MONGODB_DB_NAME]

datasets_collection = database["datasets"]


def initialize_database():
    """
    Initialize MongoDB indexes.
    """
    datasets_collection.create_index(
        [("dataset_id", ASCENDING)],
        unique=True
    )

    datasets_collection.create_index(
        [("created_at", ASCENDING)]
    )


def check_database_connection():
    """
    Check whether MongoDB is reachable.
    """
    try:
        client.admin.command("ping")
        return True
    except PyMongoError:
        return False