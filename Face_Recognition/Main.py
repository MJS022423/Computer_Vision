import torch
import argparse
import sys
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from time import time

sys.path.append(os.path.join(Path(__file__).parent, "utils.py"))
sys.path.append(os.path.join(Path(__file__).parent, "Dataset.py"))
sys.path.append(os.path.join(Path(__file__).parent, "Model.py"))
sys.path.append(os.path.join(Path(__file__).parent, "train.py"))

from utils import Earlystopper, Accuracy, collate, semi_hard_batching
from Dataset import SiameseDataset
from Model import Siamese_network
from train import train_step, valid_step

image_dir = "C:/Users/Porsha Silaroy/Documents/Marvert/image/DataSet/105_classes_pins_dataset" 

def parse_args():
  parcer = argparse.ArgumentParser()
  parcer.add_argument("--epoch", type=int, default=10,help="number of epoch default - [10]")
  parcer.add_argument("--margin", type=float, default=1.0,help="TripletLoss margin default - [1.0]")
  parcer.add_argument("--lr", type=float, default=1.0,help="Learning rate default - [0.0001]")
  args = parcer.parse_args()
  
  return args

def SplitDataset(dataset, split_size = 0.8):
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator()
    train_data, test_data = random_split(dataset, [train_size, test_size], generator)
    return train_data, test_data
   
def dataset(dataset_dir):
  dataset = SiameseDataset(dataset_dir)
  train_data, test_data = SplitDataset(dataset)
  return (train_data, test_data)

def dataLoader(dataset):
  return DataLoader(dataset, batch_size = 64, shuffle=True, collate=collate)

if __name__ == "__main__":
  
  args = parse_args()
  
  Epochs = args.epoch
  data = dataset(image_dir)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Siamese_network().to(device)
  criterion = TripletMarginLoss(margin= args.margin)
  optimizer = Adam(model.parameters(), lr = args.lr)
  scheduler = ReduceLROnPlateau(optimizer, mode= 'min', factor= 0.3, patience= 5)
  earlystop = Earlystopper(min_delta=0.01, patience=10)

  start_time = time()
  for epoch in range(Epochs):
    Train_metrics = train_step((epoch, Epochs), 
                               Accuracy, 
                               semi_hard_batching, 
                               model, 
                               dataLoader(data[0]),
                               criterion,
                               optimizer,
                               device)
    
    Test_metrics = valid_step((epoch, Epochs), 
                              Accuracy, 
                              semi_hard_batching, 
                              model, 
                              dataLoader(data[1]),
                              criterion,
                              device)
    
    if earlystop.earlystop(Test_metrics[1], model):
      earlystop.restore_best_weight(model)  
    current_lr = optimizer.param_groups[0]['lr']
    stop_time = time()
    
    print(f"Epoch {epoch+1}/{Epochs},Train Accuracy: {Train_metrics[0]:.4f} Test Loss: {Train_metrics[1]:.4f}")
    print(f"Epoch {epoch+1}/{Epochs},Test Accuracy: {Test_metrics[0]:.4f} Test Loss: {Test_metrics[1]:.4f}")
    print(f"Learning Rate: {current_lr:.6f}, Time: {stop_time}")
      


  