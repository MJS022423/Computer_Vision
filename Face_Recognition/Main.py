import torch
import argparse
import sys
import os
import torch.nn as nn
from pathlib import Path

sys.path.append(os.path.join(Path(__file__).parent, "utils.py"))
sys.path.append(os.path.join(Path(__file__).parent, "Dataset.py"))

from utils import Earlystopper
from Dataset import SiameseDataset

if __name__ == "__main__":
  parcer = argparse.ArgumentParser()
  parcer.add_argument('-Epoch', type=int, required=True)