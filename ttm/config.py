import logging
import os
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
LOG_LEVEL = logging.DEBUG

FEATURE_FOLDER = os.getenv('FEATURE_FOLDER')
DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
DEBUG_DIR = os.getenv('DEBUG_DIR')
SPLIT = os.getenv('SPLIT')
FEATURE_TYPE = os.getenv('FEATURE_TYPE')

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

ROOT = Path(__file__).resolve().parent.parent
RD_SEED = 42

MIN_PIANO_PITCH = 21
MAX_PIANO_PITCH = 108
onset_tolerance = 0.03

config = yaml.safe_load(open(f'{ROOT}/config.yaml'))  # training, dataset, general config
model_config = yaml.safe_load(open(f'{ROOT}/model_config.yaml'))  # model setup, for experiments
