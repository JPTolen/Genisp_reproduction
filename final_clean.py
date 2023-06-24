#Preprocessing final
import os

from PIL import Image
import torch
from torchvision import transforms
import rawpy 
import rawpy
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import ToTensor
import torchvision.ops as ops
import os
from torchsummary import summary
import torch.nn.functional as F
from torchvision.ops import MLP
import json
from torchvision import datasets
import pandas as pd
from torchvision.io import read_image


class ConvWB(nn.Module):
    """
    ConvWB network
    """

    def __init__(self, in_channels=3, hidden_channels=[16,32,128]):
        """
        Initialize the ConvWB network.
        Input: one or multiple images
        Returns: Input images with enhanced whitebalance obtained through
        multiplication of Input image with output of last network layer
        """
        super(ConvWB, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=7,
                                padding=3)
        self.leakyrelu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=5,
                                padding=2)
        self.leakyrelu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2],
                                kernel_size=3,
                                padding=1)
        self.leakyrelu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.adaptpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = MLP(in_channels = 128, hidden_channels = [3])



    def forward(self, input_image):


        x = self.conv1(input_image)
        x = self.leakyrelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.maxpool3(x)
        x = self.adaptpool(x)
        x = x.reshape(x.size(0),-1)

        output = self.mlp(x)

        ######### debugging ########
        # output[0][0] = 0.8
        # output[0][1] = 0.8
        # output[0][2] = 0.8
        ############################



        print('WB MATRIX VALUES ARE:', output[0])

        enhanced_image = input_image.clone()
        for i in range(len(input_image)):
          enhanced_image[i, 0, :, :] = input_image[i, 0, :, :] * output[i, 0]
          enhanced_image[i, 1, :, :] = input_image[i, 1, :, :] * output[i, 1]
          enhanced_image[i, 2, :, :] = input_image[i, 2, :, :] * output[i, 2]


        return enhanced_image

class ConvCC(nn.Module):
    """
   ConvCC network
    """

    def __init__(self, in_channels=3, hidden_channels=[16,32,128]):
        """
        Initialize the ConvCC network.
        Input: The enhanced images of the ConvWB network.
        Returns: Image with a color space that is optimal to input
        into a shallow ConvNet (next network).
        """
        super(ConvCC, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=7,
                                padding=3)
        self.leakyrelu1 = nn.LeakyReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4)
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=5,
                                padding=2)
        self.leakyrelu2 = nn.LeakyReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2],
                                kernel_size=3,
                                padding=1)
        self.leakyrelu3 = nn.LeakyReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.adaptpool = nn.AdaptiveAvgPool2d(1)

        self.mlp = MLP(in_channels = 128, hidden_channels = [9])



    def forward(self, input_image):


        x = self.conv1(input_image)
        x = self.leakyrelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.maxpool3(x)
        x = self.adaptpool(x)
        #x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0),-1)
        output = self.mlp(x)

        print('CC MATRIX VALUES ARE:', output[0])


        enhanced_image = input_image.clone()
        for i in range(len(input_image)):
          enhanced_image[i,0,:,:] = input_image[i,0,:,:]*output[i,0] + input_image[i,1,:,:]*output[i,1] + input_image[i,2,:,:]*output[i,2]
          enhanced_image[i,1,:,:] = input_image[i,0,:,:]*output[i,3] + input_image[i,1,:,:]*output[i,4] + input_image[i,2,:,:]*output[i,5]
          enhanced_image[i,2,:,:] = input_image[i,0,:,:]*output[i,6] + input_image[i,1,:,:]*output[i,7] + input_image[i,2,:,:]*output[i,8]


        return enhanced_image

class ShallowConv(nn.Module):
    """
    Shallow Convolutional network
    """

    def __init__(self, in_channels=3, hidden_channels=[16,64,3]):
        """

        """
        super(ShallowConv, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=3,
                                padding=1)
        self.instancenorm1 = nn.InstanceNorm2d(16)
        self.leakyrelu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=3,
                                padding=1)
        self.instancenorm2 = nn.InstanceNorm2d(64)
        self.leakyrelu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(hidden_channels[1], hidden_channels[2],
                                kernel_size=1,
                                padding=0)

    def forward(self, input_image):


        x = self.conv1(input_image)
        x = self.instancenorm1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.instancenorm2(x)
        x = self.leakyrelu2(x)
        x = self.conv3(x)



        return x

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


def resize_800_1333(image):
    xyz_image = image  #, raw_data
    resized_800_1333 = F.interpolate(xyz_image, size=(800, 1333), mode='bilinear')
    return resized_800_1333


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


def extract_bounding_boxes(json_file, image_folder):
    with open(json_file) as f:
        data = json.load(f)

    annotations = data['annotations']
    image_ids = set()

    # Get the unique image IDs from the annotations
    for annotation in annotations:
        image_ids.add(annotation['image_id'])
    
    bboxes = []
    category_ids = []
    image_filenames = []

    # Iterate over the images in the folder
    for annotation in annotations:
        image_id = annotation['image_id']
        image_path = os.path.join(image_folder, image_id + '.arw')

        # Check if the image file exists
        if os.path.exists(image_path):
            bbox = annotation['bbox']
            category_id = annotation['category_id']

            bboxes.append(bbox)
            category_ids.append(category_id)
            image_filenames.append(image_id + '.arw')

    return bboxes, category_ids, image_filenames


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)["annotations"]
            self.images = json.load(f)["annotations"]
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations[idx]
        img_path = os.path.join(self.img_dir, image_info["image_id"])
        # img_path = os.path.join(self.img_dir, image_info["image_id"])
        image = read_image(img_path)
        label = image_info["category_id"]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label


#defining the folders
json_path = 'RAW-NOD-main/annotations/Sony/filtered.json'
output_folder = "Test_preprocessed"
input_folder = "Test_raw"

#Getting boundingbox library
targets = []
boxes, labels, _ = extract_bounding_boxes(json_path, output_folder)
for bbox, label in zip(boxes, labels):
    # Create a dictionary with "boxes" and "labels" keys
    target = {
        "boxes": torch.tensor([bbox], dtype=torch.float32),
        "labels": torch.tensor([label], dtype=torch.int64)
    }
    targets.append(target)
# os.makedirs(output_folder, exist_ok=True)

#initializing the dataset
dataset_training = CustomImageDataset(json_path, output_folder)
# train_dataloader = DataLoader(dataset_training, batch_size=8, shuffle=True)

# Define your training DataLoader
train_loader = DataLoader(dataset_training, batch_size=4, shuffle=True)

# Create an instance of the RetinaNet model
# retinanet = models.retinanet_resnet50_fpn_v2(pretrained=True)
retinanet = models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
# Set the model to training mode
retinanet.train()

# Define the optimizer and learning rate
optimizer = optim.SGD(retinanet.parameters(), lr=0.001, momentum=0.9)

# Define the loss function (e.g., change it based on your task)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in train_loader:
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = retinanet(images)
        
        # Compute the loss
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print the loss for monitoring
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")






# print(len(dataset_training))

# epochs = 15
# batchs = 8

# # input
# # images must me resized max to 1333x800
# # for convWB and concCC resize to 256x256 using bilinear interpolation
# # RAW data must first be converted to 4 colour (RGGB)
# # data =
# # ground truth #(get labels and boxes)
# # targets = 

# # define the model
# # of the shelf retinanet with resnet50
# retinanet = models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
# output = retinanet()

# # Freeze the parameters of the backbone
# for param in retinanet.parameters():
#     param.requires_grad = False

# # our own model
# convwb_model = ConvWB()
# convcc_model = ConvCC()
# shallowconv_model = ShallowConv()

# convwb_model.train()
# convcc_model.train()
# shallowconv_model.train()


# # Two losfunctions are used
# # Regression loss = alpha balanced focal loss
# # classification loss = smooth-L1 loss
# L_reg = ops.focal_loss(input, target, alpha=None, gamma=2.0, reduction='mean')
# L_cls = nn.SmoothL1Loss()
# L_total = L_reg + L_cls

# # use adam optimazation
# optimizer = optim.Adam([convwb_model.parameters(),
#                         convcc_model.parameters(),
#                         shallowconv_model.parameters()],
#                         lr=0.01)


# # Learning rate schedule
# # learning_rate = [0.01, 0.001, 0.0001] at 5th and 10th epoch change
# lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

# # See if GPU is available + move stuff
# # mini_batch moet nog naar cuda en de optimizers ook
# if torch.cuda.is_available():
#     model.cuda()
# else:
#     pass


# # Training loop
# for epoch in range(epochs):
#   # Get the mini-batch data and targets
#   mini_batch_data = data[batch:batch + batch_size]
#   mini_batch_targets = targets[batch:batch + batch_size]

#   # Move the mini-batch data and targets to CUDA
#   mini_batch_data = mini_batch_data.cuda()
#   mini_batch_targets = mini_batch_targets.cuda()

#   # Adjust learning rate based on the schedule
#   lr_schedule.step()

#   for batch in range(0, len(data), batchs):
#     # Zero the gradients
#     optimizer.zero_grad()

#     # Forward pass
#     output = convwb_model(data)
#     output = convcc_model(output)
#     output = shallowconv_model(output)
#     output = retinanet(output)

#     ##!!!Delete later!!!##
#     # Print some stuff to check
#     print(output.size())
#     print(targets.size())

#     # Compute the individual losses
#     smooth_l1_loss_val = smooth_l1_loss(output, targets)
#     alpha_balanced_focal_loss_val = alpha_balanced_focal_loss(output, targets)

#     # Compute the total loss
#     loss_total = smooth_l1_loss_val + alpha_balanced_focal_loss_val
#     print(loss_total)

#     # Backward pass
#     loss_total.backward()

#     # Update the model parameters
#     optimizer.step()