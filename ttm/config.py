import logging
import warnings
from pathlib import Path

import yaml

LOG_LEVEL = logging.DEBUG

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

ROOT = Path(__file__).resolve().parent.parent
RD_SEED = 42

MIN_PIANO_PITCH = 21
MAX_PIANO_PITCH = 108
CHORD_N_ID = 12 * 9
onset_tolerance = 0.03

config = yaml.safe_load(open(f'{ROOT}/config.yaml'))  # training, dataset, general config
model_config = yaml.safe_load(open(f'{ROOT}/model_config.yaml'))  # model setup, for experiments
