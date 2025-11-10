import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np
from PIL import Image
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", "-i", required=True)
parser.add_argument("--save", "-s", default=None)
args = parser.parse_args()

model = models.wide_resnet50_2(pretrained=False, progress=True, num_classes=142)
model = nn.Sequential(
    model,
    nn.Sigmoid()
)
model_file = torch.load(
    '/scratch/wz3008/geoguessr/models/geoguessr_production_model.pt', 
    map_location=torch.device('cpu'))
model.load_state_dict(model_file)
model.eval()
print('Loaded Model')
import pdb; pdb.set_trace()

def reformat(arr, guess_num=1):
    num = ''
    if arr[0] >= 0.5:
        num += '-'

    arr = arr[1:]

    for idx in range(0, len(arr), 10):
        if idx == 30:
            num += '.'
        num += str(np.where(arr[idx:idx+10] == np.partition(arr[idx:idx+10].flatten(), -guess_num)[-guess_num])[0][0] % 10)

    return num

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])


img = transform(Image.open(args.image).convert("RGB"))

with torch.no_grad():
    input = img.view(-1,3,256,256)
    output = model(input)

target_split_len = int(len(output.cpu().numpy()[0])/2)
output_reformatted = reformat(output.cpu().numpy()[0][:target_split_len]) + ' ' + reformat(output.cpu().numpy()[0][target_split_len:])
output_reformatted2 = reformat(output.cpu().numpy()[0][:target_split_len]) + ' ' + reformat(output.cpu().numpy()[0][target_split_len:], 2)
print('First Guess:', output_reformatted)
print('Second Guess:', output_reformatted2)