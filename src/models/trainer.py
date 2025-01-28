import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.logger import log
import time

class Trainer:
    def __init__(self):
        # Initialize model parameters
        self.theta = None
        self.X = None
        self.y = None
        self.iterations = 0
        self.scaler = None

        # Learning rate and total iterations
        self.alpha = 0.001
        self.max_iterations = 100

        # Load and preprocess dataset
        self.load_and_preprocess_data()

        # Start with random parameters (Step 1)
        self.initialize_random_parameters()

    def load_and_preprocess_data(self):
        log("Loading and preprocessing dataset...")
        data = pd.read_csv("src/models/Hyderabad_House_Data.csv")
        data = data.dropna()

        # One-hot encoding for categorical variables
        self.X = pd.get_dummies(data.drop(['Price', 'Resale'], axis=1), drop_first=True).values
        self.y = data['Price'].values

        # Feature scaling
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # Add intercept term
        self.X = np.c_[np.ones(self.X.shape[0]), self.X]

    def initialize_random_parameters(self):
        """
        Step 1: Start with random parameters.
        """
        self.theta = np.random.rand(self.X.shape[1])

    def hypothesis(self, X):
        return np.dot(X, self.theta)

    def compute_cost(self):
        m = len(self.y)
        predictions = self.hypothesis(self.X)
        return (1 / (2 * m)) * np.sum(np.square(predictions - self.y))

    def train(self, new_parameters=None):
        """
        Train the model following the required steps.
        :param new_parameters: Parameters (weights) received from the server.
        :return: Updated parameters after a single training step.
        """
        if new_parameters is not None:
            log("Updating parameters with received values")
            self.theta = np.array(new_parameters)

        m = len(self.y)
        predictions = self.hypothesis(self.X)
        gradient = (1 / m) * np.dot(self.X.T, (predictions - self.y))
        self.theta = self.theta - self.alpha * gradient

        self.iterations += 1

        time.sleep(1)

        cost = self.compute_cost()
        log(f"Iteration {self.iterations}/{self.max_iterations}, Cost: {cost}")

        if self.iterations < self.max_iterations:
            log("Sending trained parameters to server...")
            updated_parameters = self.theta.tolist()
            return updated_parameters 

        return None 
