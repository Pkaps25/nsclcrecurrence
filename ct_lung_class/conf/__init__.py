import os
import yaml

__all__ = ["SETTINGS"]


_conf_dir = os.path.dirname(__file__)
_settings_file = os.path.join(_conf_dir, "config.yml")

with open(_settings_file, "r") as f:
    SETTINGS = yaml.safe_load(f)
