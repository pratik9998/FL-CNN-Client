import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from utils.logger import log
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import csv
import time  #Added for tracking delay time

class Trainer:
    def __init__(self):
        self.model = self.build_model()
        self.X_train, self.y_train = self.load_femnist_data()
        self.epochs_per_round = 1  # Local epochs per FL round
        self.max_fl_rounds = 10  # Total number of FL rounds
        self.current_round = 0  # Track current round
        self.final_parameters = None  
        self.send_time = None  # <-- Added to store send timestamp

        if os.path.exists("metric1.csv"):
            open("metric1.csv", "w").close()
        with open("metric1.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cost", "accuracy", "delay"])  # <-- Added "delay" column

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_femnist_data(self):
        folder_path = os.path.join(os.path.dirname(__file__), "f1278_22")  # relative path
        image_paths = glob.glob(os.path.join(folder_path, "*.png"))
        images = []
        labels = []
        for path in image_paths:
            img = Image.open(path).convert("L").resize((28, 28))  # resize to 28x28, grayscale
            images.append(np.array(img) / 255.0)  # normalize image
            label = int(os.path.basename(path).split("_")[3].split(".")[0])  # correct label extraction
            labels.append(label)

        X = np.array(images).reshape(-1, 28, 28, 1)  # reshape for CNN input
        y = np.array(labels)
        return X, y

    def train(self, new_parameters=None):
        if new_parameters is None:
            self.current_round = 1
            log(f"Starting FL round {self.current_round}/{self.max_fl_rounds}")
            self.model.fit(self.X_train, self.y_train, epochs=self.epochs_per_round, verbose=0)
            model_weights = self.model.get_weights()
            log("Local training complete. Sending model weights.")

            self.send_time = time.time()  # <-- Timestamp when sending model weights

            return model_weights
        
        # Receiving model weights (i.e., start of new round training)
        receive_time = time.time()  # <-- Timestamp on receiving model weights
        delay = None
        if self.send_time is not None:
            delay = receive_time - self.send_time  # <-- Calculate delay
            self.send_time = None  # reset after using

        self.current_round += 1
        self.model.set_weights(new_parameters)
        test_loss, test_accuracy = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        with open("metric1.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([test_loss, test_accuracy, delay])  # <-- Write delay value
        
        if self.current_round > self.max_fl_rounds:
            self.final_parameters = new_parameters
            log(f"Final model evaluation - Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
            return None
        
        log(f"Training local model for round {self.current_round}...")
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs_per_round, verbose=0)
        model_weights = self.model.get_weights()
        log("Local training complete. Sending model weights.")

        self.send_time = time.time()  # <-- Again timestamp the send time for next delay

        return model_weights
