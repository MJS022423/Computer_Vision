import torch.nn as nn
import torch.nn.functional as f
from torch.nn.functional import pairwise_distance
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights 

class Siamese_network(nn.Module):
  def __init__(self):
    super(Siamese_network, self).__init__()
    
    base_model = resnet18(weights = ResNet18_Weights.DEFAULT)
    
    for param in base_model.parameters():
      param.requires_grad = False
    
    self.feature = nn.Sequential(*list(base_model.children())[:-1])
    
    self.fc = nn.Sequential(
      nn.Dropout(0.2, inplace= True),
      nn.Linear(512,256),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(256),
      nn.Linear(256, 128)
    )
    
  def forward(self, x1, x2):
    x1 = self.forward_once(x1)
    x2 = self.forward_once(x2)
    
    dist = pairwise_distance(x1, x2)
    
    return dist
    
  def forward_once(self, x):
    
    x = self.feature(x).view(x.size(0), -1)
    return f.normalize(self.fc(x), 2, 1)