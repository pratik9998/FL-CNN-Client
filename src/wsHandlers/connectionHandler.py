import websocket
import json
from utils.logger import log

class WebSocketClient:
    def __init__(self, server_uri, trainer):
        self.server_uri = server_uri
        self.trainer = trainer
        self.ws = None

    def connect(self):
        """
        Connect to the WebSocket server.
        """
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
        log("Connection established with the server.")

    def on_message(self, ws, message):
        """
        Called when a message is received from the server.
        """
        # log(f"Message received from server: {message}")
        try:
            data = json.loads(message)

            if data["type"] == "clientId":
                log(f"Received client ID: {data['clientId']}")

            received_parameters = None

            if data["type"] == "receiveAggregatedParameters":
                received_parameters = data["parameters"]
                log("received updated parameters")
        
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
        """
        Send the parameters to the server in the specified format.
        """
        if parameters is not None:
            message = {
                "type": "sendParameters",
                "parameters": parameters
            }
            # log(f"Sending parameters to server: {message}")
            self.ws.send(json.dumps(message))
