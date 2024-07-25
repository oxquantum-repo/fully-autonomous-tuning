# There is a central database. Use this to communicate with that.


import json
import os
from datetime import datetime
import shutil


class JsonDBConnector:
    def __init__(self, exp_name, backup_dir="backups", root_dir="../data/experiments/"):
        self.db_path = os.path.join(root_dir, exp_name, "central_database")
        self.db_file_path = os.path.join(self.db_path, "db.json")
        self.backup_dir = os.path.join(self.db_path, backup_dir)
        # Initialise JSON file at specified location
        if not os.path.isfile(self.db_file_path):
            with open(self.db_file_path, "w") as db_file:
                json.dump({}, db_file)
        # Create backup directory if it does not exist
        os.makedirs(backup_dir, exist_ok=True)

    def store_value(self, keyword, value=None):
        # Backup current state before updating
        self.backup_db()
        with open(self.db_file_path, "r+") as db_file:
            data = json.load(db_file)
            # If keyword is a dictionary, merge it with the existing data.
            # Otherwise, store the keyword and its associated value.
            if isinstance(keyword, dict):
                data.update(keyword)
            else:
                data[keyword] = value
            db_file.seek(0)
            json.dump(data, db_file)
            db_file.truncate()

    def get_value(self, *keywords):
        with open(self.db_file_path, "r") as db_file:
            data = json.load(db_file)
            for keyword in keywords:
                try:
                    data = data[keyword]
                except KeyError:
                    raise KeyError(f"Keyword '{keyword}' not found in database.")
        return data

    def backup_db(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"{timestamp}_db_backup.json")
        shutil.copy2(self.db_file_path, backup_path)


if __name__ == "__main__":
    # Testing the JsonDBConnector
    path = "../data/experiments/test/central_database/db.json"
    db_connector = JsonDBConnector(
        path, "../data/experiments/test/central_database/backups"
    )

    # Assertions
    assert os.path.exists(path), "File does not exist."

    db_connector.store_value("test", "value")
    assert (
        db_connector.get_value("test") == "value"
    ), "Stored value does not match retrieved value."

    db_connector.store_value({"nested": {"key": "nested_value"}})
    assert (
        db_connector.get_value("nested", "key") == "nested_value"
    ), "Stored nested value does not match retrieved value."
