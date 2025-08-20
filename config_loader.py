import os
import yaml

# /app/config.yaml 경로 지정
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
