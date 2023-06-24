print('start')
import rawpy
print('')
import json
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import cv2
from torchvision.ops import MLP
from torch.utils.data import Dataset, DataLoader
# import pyrawide



import os
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision.ops import MLP

def preprocessing(path):
    raw = rawpy.imread(path)
    # raw_data = raw.raw_image
    raw_data = raw.raw_image.astype(np.double)

    # Normalize raw_data between 0 and 1
    black = np.reshape(np.array(raw.black_level_per_channel, dtype=np.double), (2, 2))
    black = np.tile(black, (raw_data.shape[0]//2, raw_data.shape[1]//2))
    raw_data = (raw_data - black) / (raw.white_level - black)

    # Get the color channel pattern
    n = raw.num_colors
    color_pattern = raw.raw_pattern

    red_channel = raw_data[::2, ::2]  # Extract the red channel (channel 0)
    green1_channel = raw_data[::2, 1::2]  # Extract the first green channel (channel 1)
    green2_channel = raw_data[1::2, ::2]  # Extract the second green channel (channel 2)
    blue_channel = raw_data[1::2, 1::2]  # Extract the blue channel (channel 3)

    # Averaging green
    green_avg_plane = (green1_channel + green2_channel) / 2

    # Pack the image together
    rgb_image = np.stack((red_channel, green_avg_plane, blue_channel), axis=-1)

    # Extract CST matrix
    cst = raw.rgb_xyz_matrix
    cst_selection = cst[0:3, :]

    # Convert image to XYZ
    xyz_image = rgb_image @ cst_selection.T

    xyz_image = torch.from_numpy(xyz_image).unsqueeze(0).permute(0,3,1,2)


    return xyz_image

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        # self.annotations = annotations
        # self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # annotation = self.annotations.get[idx, []]

        # Load the image and annotation
        image = preprocessing(image_path).squeeze(0)
        # with open(f'{annotation_path}') as f:
        #     data = json.load(f)
        # annotation = data

        # if self.transform:
        #     image = self.transform(image)

        return image #, annotation


with open('raw_new_Sony_RX100m7_train.json') as f:
    data = json.load(f)    
image_annotations = data['annotations']

current_directory = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)
path_to_images = os.path.join(parent_directory, 'Val_raw')


###### code for putting the image_paths in a list
###### and getting the corresponding annotations and putting
###### the annotations in a dictionary
image_paths = []
annots_list = []
for filename in os.listdir(path_to_images):
    print('filename',filename)
    f = os.path.join('..\\Val_raw', filename)
    # checking if it is a file
    image_paths.append(f)
    filename_short = ".".join(filename.split(".")[:-1])
    print('filename_short', filename_short)
    for annot in image_annotations:
        if annot['image_id'] == filename_short:
            # print(annot['image_id'])
            annots_list.append(annot)
print(image_paths)
# print(image_annots)

image_annots = {}
for item in annots_list:
    image_id = item['image_id']
    if image_id not in image_annots:
        image_annots[image_id] = []
    image_annots[image_id].append(item)

print(image_annots)


dataset = CustomDataset(image_paths,image_annots)
# print(dataset)

dataloader = DataLoader(dataset, batch_size=2)
for batch_images in dataloader:
    print(batch_images.shape)
    # print(batch_annotations)
    print('training.............................................')