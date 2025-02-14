import websocket
import json
import time
from utils.logger import log
from configure import load_config, update_client_id

class WebSocketClient:
    def __init__(self, server_uri,trainer):
        self.server_uri = server_uri
        self.ws = None
        self.trainer = trainer
        self.config = load_config()
        self.is_host = self.config["client_id"] == ""

    def connect(self):
        """Connect to the WebSocket server."""
        self.ws = websocket.WebSocketApp(
            self.server_uri,
            on_open=self.on_open,
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error
        )
        log("Connecting to server...")
        self.ws.run_forever(ping_timeout=60)

    def on_open(self, ws):
        """Handles when the WebSocket connection opens."""
        log("Connection established with the server.")

        if self.is_host:
            log("Hosting the session...")
            host_message = {
                "type": "hostSession",
                "password": self.config["client_password"],
                "total_clients": self.config["number_of_clients"]
            }
            self.ws.send(json.dumps(host_message))
        else:
            log("Joining as a non-host client...")
            join_message = {
                "type": "joinRequest",
                "client_id" : self.config["client_id"],
                "password": self.config["client_password"]
            }
            self.ws.send(json.dumps(join_message))

    def on_message(self, ws, message):
        """Handles incoming messages from the server."""
        try:
            data = json.loads(message)
            # log(f"Received message: {data}")
            received_parameters = None
            if data["type"] == "hostRegistered":
                update_client_id(data["clientId"])
                log(f"Hosting session successfully with ID {data['clientId']}")

            elif data["type"] == "clientApproved":
                log(f"Joined session successfully with ID {data['clientId']}")

            elif data["type"] == "receiveAggregatedParameters":
                received_parameters = data["parameters"]
                log("Received updated parameters from server.")

            updated_parameters = self.trainer.train(new_parameters=received_parameters)

            if updated_parameters is None:
                log("Training complete. Closing connection.")
                self.ws.close()
            else:
                self.send_parameters(updated_parameters)

        except (json.JSONDecodeError, KeyError) as e:
            log(f"Error processing server message: {e}")

    def on_close(self, ws, close_status_code, close_msg):
        log("Connection to server closed.")

    def on_error(self, ws, error):
        log(f"Error encountered: {error}")

    def send_parameters(self, parameters):
        """Send the trained parameters to the server."""
        self.config = load_config()
        if parameters is not None:
            message = {
                "client_id" : self.config["client_id"],
                "type": "sendParameters",
                "parameters": parameters
            }
            self.ws.send(json.dumps(message))
