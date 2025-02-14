import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "number_of_clients": 1,
    "client_id": "",  # Empty means it will become a host
    "client_password": "mypassword"
}

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
else :
    print("HEYY")



def load_config():
    """Load configuration from the file."""
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    """Save updated configuration to the file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def update_client_id(new_client_id):
    """Update the client_id in the configuration file."""
    config = load_config()
    config["client_id"] = new_client_id
    print
    save_config(config)
    print(f"Client hosted successfully with ID {new_client_id}")

load_config()