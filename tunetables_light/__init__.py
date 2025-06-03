import warnings
warnings.filterwarnings("ignore", message=".*CUDA initialization: CUDA unknown error.*")

from tunetables_light.scripts.transformer_prediction_interface import (
    TuneTablesZeroShotClassifier
)
from . import *
