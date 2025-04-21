import websocket
import json
from utils.logger import log
import numpy as np
import time

class WebSocketClient:
    def __init__(self, server_uri, trainer):
        self.server_uri = server_uri
        self.trainer = trainer
        self.ws = None
        self.updated_parameters = []
        self.received_parameters = []
        self.layer_shapes = []  # To store the shapes of layers
        self.current_layer_index = 0
        self.all_1d_arrays = []  # Store all 1D arrays to be sent
        self.current_array_index = 0  # Track which 1D array to send next
        self.array_to_layer_mapping = []  # Map each 1D array to its layer and position

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.server_uri,
            on_open=self.on_open,
            on_message=self.on_message,
            on_close=self.on_close,
            on_error=self.on_error
        )
        log("Connecting to server...")
        self.ws.run_forever(ping_timeout=9)

    def on_open(self, ws):
        log("Connection established with the server.")

    def on_message(self, ws, message):
        try:
            data = json.loads(message)

            if data["type"] == "clientId":
                log(f"Received client ID: {data['clientId']}")
                updated_parameters = self.trainer.train(new_parameters=None)
                if updated_parameters is None:
                    log("Training complete. Closing connection.")
                    self.ws.close()
                else:
                    # Initialize for first round
                    self.updated_parameters = updated_parameters
                    self.layer_shapes = [layer.shape for layer in self.updated_parameters]
                    self.received_parameters = []
                    
                    # Extract all 1D arrays from the parameters
                    self.all_1d_arrays = []
                    self.array_to_layer_mapping = []
                    for layer_idx, layer in enumerate(self.updated_parameters):
                        self.extract_1d_arrays(layer, layer_idx)
                    
                    self.current_array_index = 0
                    
                    # Send only the first 1D array
                    if self.all_1d_arrays:
                        self.send_parameters(self.all_1d_arrays[0])
                    else:
                        log("No parameters to send.")

            elif data["type"] == "receiveAggregatedLayer":
                # Received a parameter chunk
                received_array = np.array(data["parameters"], dtype=np.float32)
                self.received_parameters.append(received_array)
                log(f"Received 1D array {self.current_array_index + 1}/{len(self.all_1d_arrays)}")
                
                # Move to next array
                self.current_array_index += 1
                
                # If we have more arrays to send, send the next one
                if self.current_array_index < len(self.all_1d_arrays):
                    self.send_parameters(self.all_1d_arrays[self.current_array_index])
                else:
                    # All arrays have been sent and received, reconstruct and train
                    log("All 1D arrays received. Reconstructing layers...")
                    self.reconstruct_and_train()

        except Exception as e:
            log(f"Error processing server message: {e}")

    def on_close(self, ws, close_status_code, close_msg):
        log("Connection to server closed.")

    def on_error(self, ws, error):
        log(f"Error encountered: {error}")

    def send_parameters(self, parameters):
        if parameters is not None:
            message = {
                "type": "sendParameters",
                "parameters": parameters.tolist() if isinstance(parameters, np.ndarray) else parameters
            }
            print(f"Size of message: {len(json.dumps(message).encode('utf-8'))} bytes")
            self.ws.send(json.dumps(message))
            log(f"Sent 1D array {self.current_array_index + 1}/{len(self.all_1d_arrays)}")

    def extract_1d_arrays(self, array, layer_idx, indices=[]):
        """
        Recursively extract all innermost 1D arrays from a multidimensional array
        and store them in self.all_1d_arrays along with mapping information.
        """
        if len(array.shape) == 1:
            # If it's a 1D array, add it to our list with mapping info
            self.all_1d_arrays.append(array)
            self.array_to_layer_mapping.append((layer_idx, indices))
        else:
            # If it's a multi-dimensional array, iterate through first dimension
            for i in range(array.shape[0]):
                new_indices = indices + [i]
                self.extract_1d_arrays(array[i], layer_idx, new_indices)

    def reconstruct_and_train(self):
        """
        Reconstruct the layers from received 1D arrays and perform the training.
        """
        # Create empty arrays with the original shapes
        reconstructed_layers = []
        for shape in self.layer_shapes:
            reconstructed_layers.append(np.zeros(shape, dtype=np.float32))
        
        # Fill in the reconstructed layers with the received 1D arrays
        for array_idx, (received_array, (layer_idx, indices)) in enumerate(zip(self.received_parameters, self.array_to_layer_mapping)):
            # Access the target position in the layer
            target = reconstructed_layers[layer_idx]
            for idx in indices:
                target = target[idx]
            
            # Set the values
            target[:] = received_array
        
        # Now call training with the reconstructed layers
        updated_parameters = self.trainer.train(new_parameters=reconstructed_layers)
        if updated_parameters is None:
            log("Training complete. Closing connection.")
            self.ws.close()
        else:
            # Reset for next round
            self.updated_parameters = updated_parameters
            self.layer_shapes = [layer.shape for layer in self.updated_parameters]
            self.received_parameters = []
            
            # Extract all 1D arrays for next round
            self.all_1d_arrays = []
            self.array_to_layer_mapping = []
            for layer_idx, layer in enumerate(self.updated_parameters):
                self.extract_1d_arrays(layer, layer_idx)
            
            self.current_array_index = 0
            
            # Send first 1D array of next round
            if self.all_1d_arrays:
                self.send_parameters(self.all_1d_arrays[0])