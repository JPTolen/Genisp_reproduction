print('start')
import rawpy
print('')
import json
import numpy as np
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from Networks import ConvCC, ConvWB, ShallowConv
import torch.optim as optim
import torchvision.ops.focal_loss as focal_loss
import torch.nn as nn



import os
import torch
from torchsummary import summary
import torch.nn.functional as F

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
    #torch_array = torch.from_numpy(xyz_image).unsqueeze(0)
    #xyz_image = xyz_image.permute(0, 3, 1, 2)
    resized_800_1333 = F.interpolate(xyz_image, size=(256, 256), mode='bilinear')
    #resized_800_1333 = resized_800_1333.permute(0,2,3,1)
    return resized_800_1333

class CustomDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        # self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]
        

        boxes = annotation['boxes']
        labels = annotation['labels']

        # Pad the annotations to a fixed number of boxes
        max_boxes = 10  # Maximum number of boxes per image
        num_boxes = len(boxes)

        if num_boxes < max_boxes:
            # Pad the boxes and labels
            padded_boxes = boxes + [(0, 0, 0, 0)] * (max_boxes - num_boxes)
            padded_labels = labels + [0] * (max_boxes - num_boxes)
        else:
            # Trim the boxes and labels to the maximum number
            padded_boxes = boxes[:max_boxes]
            padded_labels = labels[:max_boxes]

        # Convert the padded boxes and labels to tensors
        padded_boxes = torch.tensor(padded_boxes)
        padded_labels = torch.tensor(padded_labels)

        annotation = {'boxes': padded_boxes, 'labels': padded_labels}

        #print('annotation of one image', annotation)

        # Load the image and annotation
        image = preprocessing(image_path)
        image = resize_800_1333(image).squeeze(0)
        #print('image', image.shape)

        # with open(f'{annotation_path}') as f:
        #     data = json.load(f)
        # annotation = data

        # if self.transform:
        #     image = self.transform(image)

        return image, annotation
    
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none"):
    #) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = torch.tensor(inputs)
    inputs = inputs.float()
    targets = torch.tensor(targets)
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)


    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()


    return loss



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
image_names = []
for filename in os.listdir(path_to_images):
    print('filename',filename)
    f = os.path.join('..\\Val_raw', filename)
    # checking if it is a file
    image_paths.append(f)
    filename_short = ".".join(filename.split(".")[:-1])     
    image_names.append(filename_short)
    for annot in image_annotations:
        if annot['image_id'] == filename_short:
            # print(annot['image_id'])
            annots_list.append(annot)
print(image_paths)
print(len(image_paths))
# print(image_annots)
print(image_names)
print(len(image_names))


image_annots = {}

for item in annots_list:
    image_id = item['image_id']
    if image_id not in image_annots:
        image_annots[image_id] = []

    image_annots[image_id].append(item)

annots_per_image_list = []

for name in image_names:
    print(name)
    if name not in image_annots:
        print('nottt in images')
        # targets['boxes'] = []
        # targets['labels'] = []
        annots_per_image_list.append({'boxes': [], 'labels': []})
    else:
        boxes_per_image = []
        labels_per_image = []
        for item in image_annots[name]:
            print('itemmmmmmmmmm',item)
            boxes_per_image.append(item['bbox'])
            labels_per_image.append(item['category_id'])
            print(boxes_per_image)
            # targets['boxes'] = boxes_per_image
            # targets['labels'] = labels_per_image

        annots_per_image_list.append({'boxes': boxes_per_image, 'labels': labels_per_image})
print('annots_per_image_list', len(annots_per_image_list[0]))
targets = annots_per_image_list
#print(targets)

# for item in annots_per_image_list:
#     print(item)




# print(image_annots['DSC02087'])
#print('annots_list[0]', annots_list)





# define the model
# of the shelf retinanet with resnet50
retinanet = models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
#retinanet = retinanet.cuda()

# Freeze the parameters of the backbone
for param in retinanet.parameters():
    param.requires_grad = False

# our own model
convwb_model = ConvWB()
convwb_model.to(torch.double)
convcc_model = ConvCC()
convcc_model.to(torch.double)
shallowconv_model = ShallowConv()
shallowconv_model.to(torch.double)
retinanet.to(torch.double)

# Get the parameters from each model
convwb_params = list(convwb_model.parameters())
convcc_params = list(convcc_model.parameters())
shallowconv_params = list(shallowconv_model.parameters())

# Combine the parameters into a single list
all_params = convwb_params + convcc_params + shallowconv_params

# use adam optimazation
optimizer = optim.Adam(all_params, lr=0.01)


# Learning rate schedule
# learning_rate = [0.01, 0.001, 0.0001] at 5th and 10th epoch change
lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

# See if GPU is available + move stuff
# mini_batch moet nog naar cuda en de optimizers ook
# if torch.cuda.is_available():
#     print('naar cuda gezet')
#     convwb_model.cuda()
#     convcc_model.cuda()
#     shallowconv_model.cuda()
# else:
#     pass

convwb_model.train()
convcc_model.train()
shallowconv_model.train()
retinanet.eval()

dataset = CustomDataset(image_paths,annots_per_image_list)


dataloader = DataLoader(dataset, batch_size=3)
print('DEBUGGGG')
# for batch_images, batch_annotations in dataloader:
#     print('batch_images', batch_images.shape)
#     print(batch_annotations)
#     print('training.............................................')

epochs = 15
batchs = 8
epoch_number = 0
batch_number = 0

# Training loop
for i, epoch in enumerate(range(epochs)):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Move the mini-batch data and targets to CUDA


    

    for batch_images, batch_annotations in dataloader:    #batch in range(0, len(batch_images), batchs):
        # print('BATCH {}:'.format(batch_number + 1))
        # batch_images_cuda = batch_images.cuda()
        # print(batch_images_cuda.shape)
        #batch_targets_cuda = batch_targets.cuda()
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # print(batch_images.shape)
        # print(batch_annotations)
        output = convwb_model(batch_images)
        print('output na convwv: ', output.shape)
        output = convcc_model(output)
        output = shallowconv_model(output)
        output = retinanet(output)

        ##!!!Delete later!!!##
        # Print some stuff to check
        print(output)
        print(batch_annotations)

        # Two losfunctions are used
        # Regression loss = alpha balanced focal loss
        # classification loss = smooth-L1 loss
        #L_reg = focal_loss(output, batch_annotations, alpha=None, gamma=2.0, reduction='mean')
        L_cls = nn.SmoothL1Loss(output, batch_annotations)
        L_total = L_cls #+ L_reg
        print(L_total)

        # Backward pass
        L_total.backward()

        # Update the model parameters
        optimizer.step()
        # Adjust learning rate based on the schedule
        lr_schedule.step()