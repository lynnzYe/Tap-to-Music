import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

ROOT = Path(__file__).resolve().parent.parent
RD_SEED = 52

MIN_PIANO_PITCH = 21
MAX_PIANO_PITCH = 108
