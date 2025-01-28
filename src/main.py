from wsHandlers.connectionHandler import WebSocketClient
from models.trainer import Trainer
from utils.logger import log
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    # load environment variables
    load_dotenv()
    server_uri = os.getenv("SERVER_URI")

    if not server_uri:
        log("SERVER_URI not found in .env file. Exiting.")
        exit(1)

    log("Starting client...")
    trainer = Trainer()
    client = WebSocketClient(server_uri, trainer)

    client.connect()
