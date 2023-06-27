import torch.nn as nn
from torchvision.ops import MLP

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
        
        ### Multiply the input_image with the output to obtain a white_balance enhanced_image
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

        ### Multiply the input_image with the output to obtain a color corrected enhanced_image
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