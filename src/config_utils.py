# import yaml
# from omegaconf import DictConfig, OmegaConf
# from datetime import datetime

# compiled_config = DictConfig({})  

# def add_config(cfg: DictConfig, script_name: str):
#     """Добавляет конфигурацию в глобальный объект compiled_config."""
#     global compiled_config
#     compiled_config[script_name] = cfg

# def save_compiled_config(output_path: str):
#     """Сохраняет глобальный объект compiled_config в YAML файл с добавлением текущей даты и времени."""
#     global compiled_config
#     current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     compiled_config['timestamp'] = current_datetime
#     with open(output_path, "w") as file:
#         yaml.dump(OmegaConf.to_container(compiled_config, resolve=False), file)  
import os

import yaml
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

compiled_config = DictConfig({})


def save_compiled_config(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), "loggs")
    os.makedirs(save_path, exist_ok=True)
    config_filename = f"loggs_{timestamp}.yaml"
    config_path = os.path.join(save_path, config_filename)

    # Convert OmegaConf config to dictionary and add timestamp
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['save_timestamp'] = timestamp

    # Save the updated configuration to YAML file
    with open(config_path, "w") as file:
        yaml.dump(cfg_dict, file)

    print(f"Compiled configuration saved to: {config_path}")
