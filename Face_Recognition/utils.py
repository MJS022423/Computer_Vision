import copy
import torch

class Earlystopper:
  def __init__(self, min_delta = 0.0, patience = 1):
    
    self.min_delta = min_delta
    self.patience = patience
    self.best_validation = None
    self.best_model_state = None
    self.counter = 0
    
  def earlystop(self, validation_loss, model):
    
    if self.best_validation is None:
      self.best_validation = validation_loss
      self.best_model_state = copy.deepcopy(model.state_dict())
      
    elif validation_loss <= self.best_validation - self.min_delta:
      
      self.best_validation = validation_loss
      self.best_model_state = copy.deepcopy(model.state_dict())
      self.counter = 0
    
    else:
      self.counter += 1
      
      print(f"Earlystop: {self.counter}/{self.patience}")
      if self.counter >= self.patience:
        return True
    
    return False
       
  def restore_best_weight(self, model):
    if self.best_model_state is not None:
      model.load_state_dict(self.best_model_state)
      print("Restored Best Weight")
      
def Accuracy(anchor, positive, negative, margin=0.2):
   
    d_ap = torch.norm(anchor - positive, p=2, dim=1)
    d_an = torch.norm(anchor - negative, p=2, dim=1)
    
    correct = (d_ap + margin < d_an).float()
    accuracy = correct.mean().item()
    
    return accuracy
  
def collate_fn(batch):
  batch = [item for item in batch if item[0] is not None]
  if batch is None:
    return None
  image, label = zip(*batch)
  return torch.stack(image), torch.tensor(label)

def semi_hard_batching(embedding, label, margin = 0.3):
  
  batch_size = embedding.size(0)
  dist = torch.cdist(embedding, embedding ,p=2)
  
  anchor, positive, negative = [], [], []
  
  for i in range(batch_size):
    
    mask_positive = (label == label[i])
    mask_positive[i] = False
    mask_negative = (label != label[i])
    
    if not mask_positive.any() or not mask_negative.any():
      continue
    
    pos_id = torch.where(mask_positive)[0]
    neg_id = torch.where(mask_negative)[0]
    
    d_positive = dist[i, pos_id]
    d_negative = dist[i, neg_id]
    
    mask = (d_negative.unsqueeze(1) > d_positive.unsqueeze(0)) & (d_negative.unsqueeze(1) < d_positive.unsqueeze(0) + margin)
    
    neg_i, pos_i = torch.nonzero(mask, as_tuple=True)
    for ni, pi in zip(neg_i.tolist(), pos_i.tolist()):
        anchor.append(i)
        positive.append(pos_id[pi].item())
        negative.append(neg_id[ni].item())

  if not anchor:
      empty = torch.empty(0, dtype=torch.long)
      return empty, empty, empty

  return (
      torch.tensor(anchor, dtype=torch.long),
      torch.tensor(positive, dtype=torch.long),
      torch.tensor(negative, dtype=torch.long),
  )
