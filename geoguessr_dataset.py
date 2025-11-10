import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os, os.path
from PIL import Image
from pathlib import Path
import csv
import glob

# class GeoGuessrDataset(Dataset):
#     def __init__(self, data_dir):
#         self.data_dir = data_dir
#         self.targets = np.load(os.path.join(data_dir, 'targets.npy'), allow_pickle=True)

#     def __len__(self):
#         return len(os.listdir(self.data_dir)) - 1

#     def __getitem__(self, idx):
#         data_path = os.path.join(self.data_dir, f'street_view_{idx}.jpg')
        
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#         transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.ToTensor(),
#                 normalize,
#             ])
        
#         img = pil_loader(data_path)
#         data = transform(img)
        
#         target = torch.tensor(self.targets[idx], dtype=torch.float)
        
#         return data, target
    
# def pil_loader(path: str) -> Image.Image:
#     with open(path, "rb") as f:
#         img = Image.open(f)
#         return img.convert("RGB")
    

class GeoGuessrDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = glob.glob(f"{Path(data_dir)}/images/*.png")

        target_path = os.path.join(data_dir, 'targets.npy')
        self.target_path = target_path 
        if not os.path.exists(target_path):
            mapping = {}
            with open(os.path.join(data_dir, 'sample_submission.csv'), newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_id = row["image_id"].strip()
                    lon = float(row["longitude"])
                    lat = float(row["latitude"])
                    mapping[img_id] = np.array([lon,lat],dtype=np.float32)

            np.save(target_path, mapping)
            print(f"Saved {len(mapping)} entries to {target_path}")

        obj = np.load(target_path, allow_pickle=True)
        self.targets = obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        data_path = self.image_paths[idx]
        image_id = Path(data_path).stem

        img = Image.open(data_path).convert("RGB") 
        data = self.transform(img)
        
        target = torch.tensor(self.targets[image_id])

        return data,target
