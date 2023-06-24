#Preprocessing final
import os

from PIL import Image
import torch
from torchvision import transforms
import rawpy 
import rawpy
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import pyrawide
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models


import os
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision.ops import MLP
import json



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

# def resize_800_1333(image):
#     torch_array = torch.from_numpy(image).unsqueeze(0)
#     resized_800_1333= torch_array.permute(0, 3, 1, 2)
#     resized_800_1333 = F.interpolate(resized_800_1333, size=(256, 256), mode='bilinear')
#     return resized_800_1333

def resize_800_1333(image):
    xyz_image = image  #, raw_data
    #torch_array = torch.from_numpy(xyz_image).unsqueeze(0)
    #xyz_image = xyz_image.permute(0, 3, 1, 2)
    resized_800_1333 = F.interpolate(xyz_image, size=(800, 1333), mode='bilinear')
    #resized_800_1333 = resized_800_1333.permute(0,2,3,1)
    return resized_800_1333



input_folder = "Test_raw"
output_folder = "Test_preprocessed"
os.makedirs(output_folder, exist_ok=True)

def prepare_and_store(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".ARW"):
            # Construct the input and output paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Perform preprocessing and resizing
            xyz_image = preprocessing(input_path)
            resized_image = resize_800_1333(xyz_image)

            # Save the resized image
            torch.save(resized_image, output_path)

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, folder_path, transform=None):
#         self.folder_path = folder_path
#         self.transform = transform
#         self.image_filenames = os.listdir(folder_path)

#     def __getitem__(self, index):
#         image_filename = self.image_filenames[index]
#         image_path = os.path.join(self.folder_path, image_filename)

#         # Load the resized image
#         image = torch.load(image_path)

#         # Apply the specified transform
#         if self.transform is not None:
#             image = self.transform(image)

#         return image

#     def __len__(self):
#         return len(self.image_filenames)
    
# def extract_bounding_boxes(json_file, image_folder):
#     with open(json_file) as f:
#         data = json.load(f)

#     annotations = data['annotations']
#     print(annotations)
#     image_ids = set()
#     print(image_ids)

#     # Get the unique image IDs from the annotations
#     for annotation in annotations:
#         image_ids.add(annotation['image_id'])
#     print('image_ids' , image_ids)
#     bboxes = []
#     category_ids = []


#     # Iterate over the images in the folder
#     for image_id in image_ids:
#         image_id = str(image_id)
#         #print(type(image_id))
#         image_path = os.path.join(image_folder, image_id + '.arw')
#         print(image_path)

#         # Check if the image file exists
#         if os.path.exists(image_path):
#             # Get the annotations for the current image ID
#             image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image_id]

#             # Extract the bounding boxes and category IDs
#             for annotation in image_annotations:
#                 bbox = annotation['bbox']
#                 category_id = annotation['category_id']
#                 print('image_name:', image_nombre)

#                 bboxes.append(bbox)
#                 category_ids.append(category_id)

#     return bboxes, category_ids

def extract_bounding_boxes(json_file, image_folder):
    with open(json_file) as f:
        data = json.load(f)

    annotations = data['annotations']
    print(annotations)
    image_ids = set()
    print(image_ids)

    # Get the unique image IDs from the annotations
    for annotation in annotations:
        image_ids.add(annotation['image_id'])
    print('image_ids', image_ids)
    
    bboxes = []
    category_ids = []
    image_filenames = []

    # Iterate over the images in the folder
    for annotation in annotations:
        image_id = annotation['image_id']
        image_path = os.path.join(image_folder, image_id + '.arw')
        print(image_path)

        # Check if the image file exists
        if os.path.exists(image_path):
            bbox = annotation['bbox']
            category_id = annotation['category_id']

            bboxes.append(bbox)
            category_ids.append(category_id)
            image_filenames.append(image_id + '.arw')

    return bboxes, category_ids, image_filenames

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, json_file, transform=None):
        self.folder_path = folder_path
        self.json_file = json_file
        self.transform = transform
        self.bboxes, self.category_ids, self.image_nombres = extract_bounding_boxes(json_file, folder_path)
        self.image_filenames = [image_nombre  for image_nombre in self.image_nombres]

    def __getitem__(self, index):
        print(self.image_filenames)
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.folder_path, image_filename)
        print('json' ,self.json_file)
        # Load the resized image
        image = torch.load(image_path)

        # Get the corresponding bounding box for the image
        bbox = self.bboxes[index]

        # Apply the specified transform to the image
        # if self.transform is not None:
        #     image = self.transform(image)

        return image, bbox

    def __len__(self):
        return len(self.image_filenames)
    
# # Define the transformation(s) to apply to the images
# transform = transforms.ToTensor()

# # Create the custom dataset instance
# dataset = CustomDataset(output_folder, transform=transform)

# # Create the DataLoader
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
epochs = 15
batch_size = 8
json_path = 'RAW-NOD-main/annotations/Sony/Raw_json_train.json'

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize((800, 1333)),  # Resize to maximum size
    transforms.ToTensor()
])

# Create the custom dataset instance
# dataset = CustomDataset(output_folder, transform=transform)
dataset = CustomDataset(output_folder, json_path, transform=transform)
# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
retinanet = models.detection.retinanet_resnet50_fpn(pretrained=True)
retinanet.train()

# Define the loss functions
smooth_l1_loss = nn.SmoothL1Loss()
# alpha_balanced_focal_loss = AlphaBalancedFocalLoss(alpha=0.5, gamma=2)

# Define the optimizer
optimizer = optim.Adam(retinanet.parameters(), lr=0.01)

# Learning rate schedule
lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

targets = []

boxes, labels, _ = extract_bounding_boxes(json_path, output_folder)

# for bbox, label in zip(boxes, labels):
#     # Convert bbox and label to tensors
#     bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
#     label_tensor = torch.tensor(label, dtype=torch.long)

#     # Concatenate bbox and label tensors along dim 0
#     target = torch.cat((bbox_tensor, label_tensor.unsqueeze(0)), dim=0)
#     targets.append(target)

# targets_tensor = torch.stack(targets)

for bbox, label in zip(boxes, labels):
    # Create a dictionary with "boxes" and "labels" keys
    target = {
        "boxes": torch.tensor([bbox], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64)
    }
    targets.append(target)



# Training loop
for epoch in range(epochs):
    # Adjust learning rate based on the schedule
    lr_schedule.step()

    for images, bboxes in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = retinanet(images, targets)

        # Compute the individual losses
        # smooth_l1_loss_val = smooth_l1_loss(outputs, bboxes)
        # alpha_balanced_focal_loss_val =  alpha_balanced_focal_loss(outputs, bboxes)

        # Compute the total loss
        # loss_total = smooth_l1_loss_val #+ alpha_balanced_focal_loss_val
        # print(loss_total.item())
        print('outputs', outputs)

        # Backward pass
        # loss_total.backward()

        # Update the model parameters
        # optimizer.step()