from pymongo import MongoClient
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

# Load .env from the same directory as this file so it works regardless of cwd
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Null-object helpers — used when DB is disabled so attribute access on
# db_config.db.collection.method() never raises AttributeError.
# ---------------------------------------------------------------------------

class _NullCursor:
    """Mimics a PyMongo cursor with no documents."""
    def __iter__(self):
        return iter([])
    def sort(self, *a, **kw):
        return self
    def limit(self, *a, **kw):
        return self
    def skip(self, *a, **kw):
        return self


class _NullCollection:
    """Returns safe no-op / empty responses for every collection access."""
    def find(self, *a, **kw):
        return _NullCursor()
    def find_one(self, *a, **kw):
        return None
    def insert_one(self, *a, **kw):
        return None
    def update_one(self, *a, **kw):
        return None
    def delete_one(self, *a, **kw):
        return None
    def count_documents(self, *a, **kw):
        return 0
    def distinct(self, *a, **kw):
        return []
    def create_index(self, *a, **kw):
        return None
    # aggregate / bulk helpers
    def aggregate(self, *a, **kw):
        return _NullCursor()


class _NullDatabase:
    """Returns a _NullCollection for every attribute access."""
    def __getattr__(self, name):
        return _NullCollection()
    def list_collection_names(self):
        return []
    def create_collection(self, *a, **kw):
        return _NullCollection()
    def __getitem__(self, name):
        return _NullCollection()


class MongoDBConfig:
    _DEFAULT_URI = 'mongodb://localhost:27017/'

    def __init__(self):
        # Read connection settings from environment variables.
        # Falls back to localhost so the app works out-of-the-box without a .env file.
        raw_uri = os.getenv('MONGO_URI')
        if raw_uri:
            self.mongo_uri = raw_uri
        else:
            self.mongo_uri = self._DEFAULT_URI
            print(
                "[CONFIG] WARNING: MONGO_URI is not set in the environment. "
                f"Defaulting to {self._DEFAULT_URI}. "
                "Create a .env file with MONGO_URI=<your-uri> to override."
            )
            logger.warning("MONGO_URI not set — using default: %s", self._DEFAULT_URI)

        self.database_name = os.getenv('DB_NAME', 'Passenger_Anomaly')
        self.client = None
        self.db_disabled = False

        # _db is the real or null database; accessed via the .db property below
        self._db = _NullDatabase()

    # ------------------------------------------------------------------
    # Public property — always safe to access
    # ------------------------------------------------------------------
    @property
    def db(self):
        return self._db

    def connect(self):
        """Establish MongoDB connection with retry logic"""
        if self.db_disabled:
            print("[CONFIG] DB-disabled mode active — skipping connection attempt.")
            return False
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Will throw exception if connection fails
            self._db = self.client[self.database_name]
            print(f"[CONFIG] Connected to MongoDB: {self.database_name}")

            # Create collections with validation schemas
            self._create_collections()
            self._create_indexes()

            return True
        except Exception as e:
            err = str(e)
            print(f"[CONFIG] MongoDB connection failed: {err}")
            if 'ECONNREFUSED' in err or 'Connection refused' in err or '27017' in err:
                print(
                    "[CONFIG] HINT: MongoDB is not running on this machine. "
                    "Start it with:  brew services start mongodb-community"
                    "  — or check the MongoDB installation section in README."
                )
            logger.error("MongoDB connection failed: %s", err)
            self.db_disabled = True  # fall back to null mode after failure
            return False

    def test_db_connection(self):
        """
        Ping MongoDB and log success/failure.
        Call this once at application startup after connect().
        Returns True if the server is reachable, False otherwise.
        """
        if self.db_disabled or self.client is None:
            print("[CONFIG] test_db_connection: DB-disabled mode — ping skipped.")
            logger.warning("test_db_connection: skipped (DB-disabled mode).")
            return False
        try:
            self.client.admin.command('ping')
            print("[CONFIG] test_db_connection: MongoDB ping successful ✓")
            logger.info("MongoDB ping successful.")
            return True
        except Exception as e:
            print(f"[CONFIG] test_db_connection: MongoDB ping FAILED — {e}")
            logger.error("MongoDB ping failed: %s", e)
            return False

    def _create_collections(self):
        """Create collections with schema validation"""
        collections = self.db.list_collection_names()

        # passengers collection
        if 'passengers' not in collections:
            self.db.create_collection('passengers', validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['passenger_id', 'created_at'],
                    'properties': {
                        'passenger_id': {'bsonType': 'string'},
                        'name': {'bsonType': 'string'},
                        'total_journeys': {'bsonType': 'int', 'minimum': 0},
                        'last_seen': {'bsonType': 'date'},
                        'created_at': {'bsonType': 'date'}
                    }
                }
            })

        # journeys collection
        if 'journeys' not in collections:
            self.db.create_collection('journeys', validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['journey_id', 'bus_turn_id', 'passenger_id', 'entrance_time'],
                    'properties': {
                        'journey_id': {'bsonType': 'string'},
                        'bus_turn_id': {'bsonType': 'string'},
                        'passenger_id': {'bsonType': 'string'},
                        'entrance_time': {'bsonType': 'date'},
                        'exit_time': {'bsonType': 'date'},
                        'travel_time_seconds': {'bsonType': 'int', 'minimum': 0},
                        'date': {'bsonType': 'date'}
                    }
                }
            })

        # images collection
        if 'images' not in collections:
            self.db.create_collection('images', validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['image_id', 'passenger_id', 'image_type', 'timestamp'],
                    'properties': {
                        'image_id': {'bsonType': 'string'},
                        'passenger_id': {'bsonType': 'string'},
                        'image_type': {'bsonType': 'string', 'enum': ['entrance', 'exit']},
                        'image_path': {'bsonType': 'string'},
                        'timestamp': {'bsonType': 'date'},
                        'journey_id': {'bsonType': 'string'}
                    }
                }
            })

        # alerts collection
        if 'alerts' not in collections:
            self.db.create_collection('alerts', validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['alert_id', 'passenger_id', 'alert_type', 'timestamp'],
                    'properties': {
                        'alert_id': {'bsonType': 'string'},
                        'passenger_id': {'bsonType': 'string'},
                        'journey_id': {'bsonType': 'string'},
                        'alert_type': {'bsonType': 'string', 'enum': ['anomaly', 'normal']},
                        'confidence': {'bsonType': 'double', 'minimum': 0, 'maximum': 1},
                        'timestamp': {'bsonType': 'date'},
                        'image_paths': {'bsonType': 'array', 'items': {'bsonType': 'string'}}
                    }
                }
            })

        # system_logs collection
        if 'system_logs' not in collections:
            self.db.create_collection('system_logs', validator={
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['log_id', 'event_type', 'timestamp'],
                    'properties': {
                        'log_id': {'bsonType': 'string'},
                        'event_type': {'bsonType': 'string'},
                        'description': {'bsonType': 'string'},
                        'timestamp': {'bsonType': 'date'},
                        'metadata': {'bsonType': 'object'}
                    }
                }
            })

    def _create_indexes(self):
        """Create indexes for faster queries"""
        # passengers collection indexes
        self.db.passengers.create_index([('passenger_id', 1)], unique=True)
        self.db.passengers.create_index([('last_seen', -1)])

        # journeys collection indexes
        self.db.journeys.create_index([('journey_id', 1)], unique=True)
        self.db.journeys.create_index([('passenger_id', 1)])
        self.db.journeys.create_index([('bus_turn_id', 1)])
        self.db.journeys.create_index([('date', -1)])

        # images collection indexes
        self.db.images.create_index([('image_id', 1)], unique=True)
        self.db.images.create_index([('passenger_id', 1)])
        self.db.images.create_index([('timestamp', -1)])

        # alerts collection indexes
        self.db.alerts.create_index([('alert_id', 1)], unique=True)
        self.db.alerts.create_index([('passenger_id', 1)])
        self.db.alerts.create_index([('timestamp', -1)])
        self.db.alerts.create_index([('alert_type', 1)])

        # system_logs collection indexes
        self.db.system_logs.create_index([('log_id', 1)], unique=True)
        self.db.system_logs.create_index([('timestamp', -1)])
        self.db.system_logs.create_index([('event_type', 1)])

    def get_collection(self, collection_name):
        """Get a specific collection"""
        return self.db[collection_name]

    def log_event(self, event_type, description, metadata=None):
        """Log system event. Silently skips the write in DB-disabled mode."""
        if self.db_disabled:
            logger.debug("log_event skipped (DB-disabled): %s — %s", event_type, description)
            return
        log_entry = {
            'log_id': f'log_{datetime.now().strftime("%Y%m%d%H%M%S%f")}',
            'event_type': event_type,
            'description': description,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.db.system_logs.insert_one(log_entry)

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()


# Global MongoDB instance
db_config = MongoDBConfig()