import torch
import argparse
import sys
import os
import torch.nn as nn
from pathlib import Path
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(Path(__file__).parent, "utils.py"))
sys.path.append(os.path.join(Path(__file__).parent, "Dataset.py"))
sys.path.append(os.path.join(Path(__file__).parent, "Model.py"))
sys.path.append(os.path.join(Path(__file__).parent, "train.py"))

from utils import Earlystopper, Accuracy, collate
from Dataset import SiameseDataset
from Model import Siamese_network
from train import train_step, valid_step

def parse_args():
  parcer = argparse.ArgumentParser()  
  args = parcer.parse_args()
  
  return args

if __name__ == "__main__":
  
  args = parse_args()
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  Epochs = args.Epoch
  model = Siamese_network().to(device)
  criterion = TripletMarginLoss(margin= args.margin)
  optimizer = Adam(model.parameters(), lr = args.lr)
  scheduler = ReduceLROnPlateau(optimizer, mode= 'min', factor= 0.3, patience= 5)
  earlystop = Earlystopper(min_delta= 0.01, patience= 10)
  
  for epoch in range(Epochs):
    
    train_step((epoch, Epochs), )
    valid_step()

  