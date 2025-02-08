import os
import sys
import pymongo
import certifi

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    """
    
    # Shared MongoClient instance across all MongoDBClient instances
    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes a connection to the MongoDB database. If an existing connection is not found, a new one is established.
        
        Parameters:
        ----------
        database_name : str, optional
            Specifies the name of the MongoDB database to connect to. The default is defined by the DATABASE_NAME constant.
        
        Raises:
        ------
        MyException
            Occurs when there is an issue connecting to MongoDB or when the environment variable for the MongoDB URL is not set.
        """
        try:
            # Checks whether a MongoDB client connection already exists; if not, creates a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)  # Retrieves the MongoDB URL from environment variables
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' is not set.")

                # Establishes a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

            # Uses the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection has been successfully established")
        except Exception as e:
            # Raises a custom exception with traceback details if the connection fails
            raise MyException(e, sys)
