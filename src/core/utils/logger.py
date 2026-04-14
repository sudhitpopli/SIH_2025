import csv
import os
import time

class CSVLogger:
    def __init__(self, log_dir, config_name="default"):
        self.log_dir = log_dir
        self.config_name = config_name
        self.file_path = os.path.join(log_dir, "training_log.csv")
        
        # Ensure log dir exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Initialize file with headers if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Config_ID", "Episode", "Avg_Reward", "Loss", "Epsilon"])

    def log_episode(self, episode, avg_reward, loss, epsilon):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self.config_name,
                episode,
                f"{avg_reward:.4f}",
                f"{loss:.4f}",
                f"{epsilon:.4f}"
            ])
