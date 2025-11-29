import logging
import os
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

LOG_LEVEL = logging.DEBUG
load_dotenv(os.path.join(Path(__file__).resolve().parent, ".env"))
dotenv_config = {key: os.getenv(key) for key in [
    "FEATURE_FOLDER", "FEATURE_TYPE", "SPLIT", "DATA_DIR", "TRAIN_NAME", "OUTPUT_DIR",
    "DEBUG_DIR", "CKPT_PATH", "MAESTRO_PATH", "ASAP_PATH", "POP909_PATH", "HANNDS_PATH"
]}

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

ROOT = Path(__file__).resolve().parent.parent
RD_SEED = 42

MIN_PIANO_PITCH = 21
MAX_PIANO_PITCH = 108
onset_tolerance = 0.03

config = yaml.safe_load(open(f'{ROOT}/config.yaml'))  # training, dataset, general config
model_config = yaml.safe_load(open(f'{ROOT}/model_config.yaml'))  # model setup, for experiments
