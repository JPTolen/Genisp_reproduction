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

def scale_bbox(bbox, original_size, target_size):
    # Calculate scaling factors for width and height
    width_scale = target_size[1] / original_size[1]
    height_scale = target_size[0] / original_size[0]

    # Scale the bounding box coordinates
    scaled_bbox = torch.FloatTensor([
        bbox[0] * width_scale,                     # x_min
        bbox[1] * height_scale,                    # y_min
        (bbox[0] + bbox[2]) * width_scale,         # x_max
        (bbox[1] + bbox[3]) * height_scale         # y_max
    ])

    return scaled_bbox

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
        max_boxes = 0
        for item in annots_per_image_list:
            boxes_count = len(item['boxes'])
            if boxes_count > max_boxes:
                max_boxes = boxes_count
        self.max_boxes = max_boxes
        #print('max_boxes', max_boxes)
        # self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation = self.annotations[idx]

        boxes_scaled = []
        boxes = annotation['boxes']
        #print('boxes_get_item', boxes)
        for box in boxes:
            box = scale_bbox(box, original_size=(5496, 3672), target_size=(256, 256))
            boxes_scaled.append(box)
        boxes = boxes_scaled
        # boxes_lists = [[int(round(coord)) for coord in box.tolist()] for box in boxes]
        boxes_lists = [[coord for coord in box.tolist()] for box in boxes]
        boxes = boxes_lists

        labels = annotation['labels']

        # Pad the annotations to a fixed number of boxes
        max_boxes = self.max_boxes  # Maximum number of boxes per image
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

        # Load the image
        image = preprocessing(image_path)
        image = resize_800_1333(image).squeeze(0)


        return image, annotation


def process_resnet_output(annot_length, batch_images, batch_annotations, output):
    padded_outputs = []
    tensor1 = torch.zeros((1,annot_length)) #placeholder for annots
    tensor2 = torch.zeros((1,annot_length)) #placeholder for scores
    tensor3 = torch.zeros((1,annot_length,4)) #placeholder for boxes
    for i in range(len(batch_images)):
        output_sized = output[i] #.unsqueeze(0)
        #print('output_sized',output_sized)

        padded_boxes = torch.nn.functional.pad(output_sized['boxes'], pad=(0, 0, 0, annot_length - len(output_sized['boxes'])), mode='constant', value=0)
        padded_scores = torch.nn.functional.pad(output_sized['scores'], pad=(0, annot_length - len(output_sized['scores'])), mode='constant', value=0)
        padded_labels = torch.nn.functional.pad(output_sized['labels'], pad=(0, annot_length - len(output_sized['labels'])), mode='constant', value=0)
        padded_output = {'boxes': padded_boxes, 'scores': padded_scores, 'labels': padded_labels}
        #print('padded_output', padded_output)
        padded_outputs.append(padded_output)

        annot_labels = batch_annotations['labels'][i,:]
        batch_annotations_y = torch.eq(annot_labels, padded_labels).int().unsqueeze(0)
        #print(batch_annotations_y.shape)

        tensor1 = torch.cat((tensor1, batch_annotations_y), dim=0)
        tensor2 = torch.cat((tensor2, padded_scores.unsqueeze(0)), dim=0)

        padded_boxes = padded_boxes.unsqueeze(0)
        annot_boxes = batch_annotations['boxes'][i,:,:]
        tensor3 = torch.cat((tensor3, padded_boxes), dim=0)


    output_boxes = tensor3[1:,:,:]
    annot_boxes = batch_annotations['boxes']
    batch_scores = tensor2[1:]
    batch_annotations_y = tensor1[1:]

    return output_boxes, annot_boxes, batch_scores, batch_annotations_y

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

    inputs = inputs.clone().detach().requires_grad_(True)
    inputs = inputs.float()
    targets = targets.float()
    targets = targets.clone().detach().requires_grad_(True)

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


def process_gt_annotations(path_to_images, image_annotations):
    ''' code for putting the image_paths in a list
        and getting the corresponding annotations and putting
        the annotations in a dictionary
        
        Constructing a list with annotations per image
        annots_per_image_list:
        [   {'boxes': ... , 'labels': ...},
            { ... } ,
            {'boxes': ... , 'labels': ...}  ]
    '''

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

    image_annots = {}

    for item in annots_list:
        image_id = item['image_id']
        if image_id not in image_annots:
            image_annots[image_id] = []
        image_annots[image_id].append(item)
    annots_per_image_list = []
    for name in image_names:
        if name not in image_annots:
            print('nottt in images')
            # targets['boxes'] = []
            # targets['labels'] = []
            annots_per_image_list.append({'boxes': [], 'labels': []})
        else:
            boxes_per_image = []
            labels_per_image = []
            for item in image_annots[name]:
                boxes_per_image.append(item['bbox'])
                labels_per_image.append(item['category_id'])
                # targets['boxes'] = boxes_per_image
                # targets['labels'] = labels_per_image
            annots_per_image_list.append({'boxes': boxes_per_image, 'labels': labels_per_image})
    print('annots_per_image_list', annots_per_image_list)

    return image_paths, annots_list, image_names, annots_per_image_list

def get_max_boxes(annots_per_image_list):

    max_boxes = 0
    for item in annots_per_image_list:
        boxes_count = len(item['boxes'])
        if boxes_count > max_boxes:
            max_boxes = boxes_count
    print('max_boxes', max_boxes)
    return max_boxes



##### STARTING THE CODE

###############################################
#### choose between 'train' or 'test' #########
###############################################
network_mode = 'train'

### Get ground truth image annotations
with open('raw_new_Sony_RX100m7_train.json') as f:
    data = json.load(f)    
image_annotations = data['annotations']

current_directory = os.getcwd()

### Get the image paths
### Change if image directory is stored locally
parent_directory = os.path.dirname(current_directory)
path_to_images = os.path.join(parent_directory, 'Val_raw')

### Get ground truth annotations
### in the correct format
image_paths, annots_list, image_names, annots_per_image_list = process_gt_annotations(path_to_images, 
                                                                                      image_annotations)

### Define the detector
### Using of-the-shelf retinanet with resnet50
retinanet = models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
#retinanet = retinanet.cuda()

### Freeze the parameters of the backbone
for param in retinanet.parameters():
    param.requires_grad = False

### Initialize the conv nets
convwb_model = ConvWB()
convwb_model.to(torch.double)
convcc_model = ConvCC()
convcc_model.to(torch.double)
shallowconv_model = ShallowConv()
shallowconv_model.to(torch.double)
retinanet.to(torch.double)

### Get the parameters from each model
convwb_params = list(convwb_model.parameters())
convcc_params = list(convcc_model.parameters())
shallowconv_params = list(shallowconv_model.parameters())

### Combine the parameters into a single list
all_params = convwb_params + convcc_params + shallowconv_params

### use adam optimazation
optimizer = optim.Adam(all_params, lr=0.01)


### Learning rate schedule

lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

### Option for using GPU
# See if GPU is available + move stuff
# mini_batch moet nog naar cuda en de optimizers ook
# if torch.cuda.is_available():
#     print('naar cuda gezet')
#     convwb_model.cuda()
#     convcc_model.cuda()
#     shallowconv_model.cuda()
# else:
#     pass

### Set networks to either test or train
if network_mode == 'train':
    convwb_model.train()
    convcc_model.train()
    shallowconv_model.train()
elif network_mode == 'test':
    convwb_model.eval()
    convcc_model.eval()
    shallowconv_model.eval()

    ### load model parameters for test mode
    convwb_model.load_state_dict(torch.load('convwb_model.pth'))
    convcc_model.load_state_dict(torch.load('convcc_model.pth'))
    shallowconv_model.load_state_dict(torch.load('shallowconv_model.pth'))
retinanet.eval()



### Determine the maximum amount of boxes per 
### image in the annotations 
max_boxes = get_max_boxes(annots_per_image_list)

batch_size = 3
dataset = CustomDataset(image_paths,annots_per_image_list, max_boxes)
dataloader = DataLoader(dataset, batch_size)



### debugging dataloaders
# for batch_images, batch_annotations in dataloader:
#     print('batch_images', batch_images.shape)
#     print(batch_annotations)
#     print('training.............................................')





regression_loss = []
classification_loss = []
epochs = 3

### Training loop
for i, epoch in enumerate(range(epochs)):
    
    print('EPOCH', i+1)

    for batch_images, batch_annotations in dataloader:
        ### Options for GPU
        #batch_targets = batch_targets.cuda()

        ### Zero the gradients
        optimizer.zero_grad()

        ### Forward pass
        print(batch_images.shape)
        if network_mode == 'train':
            batch_images.requires_grad = True
        elif network_mode == 'test':
            batch_images.requires_grad = False
        
        output = convwb_model(batch_images)
        output = convcc_model(output)
        output = shallowconv_model(output)
        output = retinanet(output)
        
        ### Determine annotation length per batch
        ### Number of GT boxes per image 
        annot_length = batch_annotations['labels'].shape[1]

        output_boxes, annot_boxes, batch_scores, batch_annotations_y = process_resnet_output(annot_length, 
                                                                                             batch_images, 
                                                                                             batch_annotations, 
                                                                                             output)


        ### Calculate losses
        L_cls = sigmoid_focal_loss(batch_scores, batch_annotations_y, reduction='mean')
        regression_loss.append(L_cls.item())

        smoothl1loss = nn.SmoothL1Loss()
        L_reg = smoothl1loss(output_boxes, annot_boxes)
        classification_loss.append(L_reg.item())

        L_total = L_reg + L_cls

        print('regression_loss', regression_loss)
        print('classification_loss', classification_loss)

        ### Backward pass
        L_total.backward()

        ### Update the model parameters
        optimizer.step()

        ###Adjust learning rate based on the schedule
        lr_schedule.step()

### Save the model parameters after training
if network_mode == 'train':
    # Save the parameters of the models
    torch.save(convwb_model.state_dict(), 'convwb_model.pth')
    torch.save(convcc_model.state_dict(), 'convcc_model.pth')
    torch.save(shallowconv_model.state_dict(), 'shallowconv_model.pth')