import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
  def __init__(self, image_dir=None, transform = None):
    
    self.image_dir = image_dir
    self.transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(13),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.02),
        transforms.RandomAffine(degrees=20, translate=(0.05, 0.05)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
      ])
    
    self.image = self.prepare_triplet_data(self.image_dir)
  
  def prepare_triplet_data(self, image_dir_path):
    
    if image_dir_path:
    
      image_path = []
      
      for idx, folder in enumerate(os.listdir(image_dir_path)):
        Image_dir = os.path.join(image_dir_path, folder)
        if os.path.isdir(Image_dir):
          for file in os.listdir(Image_dir):
            if file.lower().endswith(("jpg","png")):
              image_full_path = os.path.join(Image_dir, file)
              image_path.append((image_full_path, idx))
            
    return image_path
        
  def __len__(self):
    return len(self.image)
  
  def __getitem__(self, index):
        
    image, label = self.image[index]
    try:
      image = Image.open(image).convert("RGB")
    except Exception as e:
      return None, None
    if image is None:
      return None, None
    image = self.transform(image)
    if torch.isnan(image).any():
      print(f"Nan Detected in image at index {index}")
      return None, None
      
    return image, label