def log(message):
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {message}")
