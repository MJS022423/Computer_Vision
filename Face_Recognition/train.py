import torch.nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import pairwise_distance


def train_step(Epoch, Accuracy, Mining, Model: torch.nn.Module, 
               DataLoader: DataLoader, 
               Criterion: torch.nn.Module, 
               Optimizer: torch.optim.Optimizer,
               Device: torch.Device):
  
  Model.train()
  
  for x_train in tqdm(DataLoader, total= len(DataLoader), desc = f"Epoch: {Epoch[0] + 1} / {Epoch[1]} - Training:", leave = False):
    
    img, label = [data.to(Device) for data in x_train]
    
    embedding = Model.forward_once(img)
    
    a, p, n = Mining(embedding, label)
    
    if a.numel() == 0:
      continue
    
    a = embedding[a]
    p = embedding[p]
    n = embedding[n]
    
    dist_A = pairwise_distance(a, p)
    dist_B = pairwise_distance(a, n)
    
    loss = Criterion(a, p, n)
    acc = Accuracy(a, p, n)
    
    running_loss += loss.item()
    running_acc += acc
    
    Optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
    Optimizer.step()
    
  return running_acc, running_loss

def valid_step(Epoch, Accuracy, Mining, Model: torch.nn.Module, 
                    DataLoader: DataLoader, 
                    Criterion: torch.nn.Module, 
                    Device: torch.Device):
  Model.eval()
  with torch.inference_mode():
    for x_test in tqdm(DataLoader, total= len(DataLoader), desc = f"Epoch: {Epoch[0] + 1} / {Epoch[1]} - Training:", leave = False):
      
      img, label = [data.to(Device) for data in x_test]
      
      embedding = Model.forward_once(img)
      
      a, p, n = Mining(embedding, label)
      
      if a.numel() == 0:
        continue
      
      a = embedding[a]
      p = embedding[p]
      n = embedding[n]
      
      dist_A = pairwise_distance(a, p)
      dist_B = pairwise_distance(a, n)
      
      loss = Criterion(a, p, n)
      acc = Accuracy(a, p, n)
      
      running_loss += loss.item()
      running_acc += acc
      
    return running_acc, running_loss
  