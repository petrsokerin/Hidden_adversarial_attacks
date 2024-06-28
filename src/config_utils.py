# config_utils.py
import yaml
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

compiled_config = DictConfig({})  # Глобальный объект для объединения конфигураций

def add_config(cfg: DictConfig, script_name: str):
    """Добавляет конфигурацию в глобальный объект compiled_config."""
    global compiled_config
    compiled_config[script_name] = cfg

def save_compiled_config(output_path: str):
    """Сохраняет глобальный объект compiled_config в YAML файл с добавлением текущей даты и времени."""
    global compiled_config
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    compiled_config['timestamp'] = current_datetime
    with open(output_path, "w") as file:
        yaml.dump(OmegaConf.to_container(compiled_config, resolve=True), file)
