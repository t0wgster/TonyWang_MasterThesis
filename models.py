#torch
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, random_split
from torch.cuda.amp import GradScaler
#from torchvision.transforms import v2
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

##############################################
############## UNET CLASSIC ##################
##############################################

class encoding_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU())
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU())
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x) 

class encoding_block_gelu_2_conv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block_gelu_2_conv,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Dropout(p=0.15))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)  
    
class encoding_block_gelu_3_conv(nn.Module):
    def __init__(self,in_channels, mid_channels, out_channels):
        super(encoding_block_gelu_3_conv,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(mid_channels))
        model.append(nn.GELU())
        model.append(nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Dropout(p=0.15))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)  

class unet_model_classic(nn.Module):
    def __init__(self,out_channels,features=[64, 128, 256, 512]):
        super(unet_model_classic,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(3,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

##############################################
################ UNET GELU ###################
##############################################

class encoding_block_gelu(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(encoding_block_gelu,self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.GELU())
        model.append(nn.Dropout(p=0.15))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        return self.conv(x)  

class unet_model_gelu(nn.Module):
    def __init__(self,out_channels,features=[64, 128, 256, 512]):
        super(unet_model_gelu,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block_gelu(3,features[0])
        self.conv2 = encoding_block_gelu(features[0],features[1])
        self.conv3 = encoding_block_gelu(features[1],features[2])
        self.conv4 = encoding_block_gelu(features[2],features[3])
        self.conv5 = encoding_block_gelu(features[3]*2,features[3])
        self.conv6 = encoding_block_gelu(features[3],features[2])
        self.conv7 = encoding_block_gelu(features[2],features[1])
        self.conv8 = encoding_block_gelu(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block_gelu(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

      
########################################################
################ UNET Resnet Backbone ##################
########################################################

resnet = torchvision.models.resnet.resnet50(pretrained=True)

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x

############################################
############### HSI UNET ###################
############################################

class preprocessing_block(nn.Module):
    def __init__(self, channels_3):
        super(preprocessing_block,self).__init__()
        model = []
        model.append(nn.Conv2d(45, 30, 1, 1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(30))
        model.append(nn.GELU())
        model.append(nn.Conv2d(30, 15, 1, 1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(15))
        model.append(nn.GELU())
        model.append(nn.Conv2d(15, 10, 1, 1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(10))
        model.append(nn.GELU())
        if channels_3==3:
            model.append(nn.Conv2d(10, 3, 1, 1, padding=0, bias=False))
            model.append(nn.BatchNorm2d(3))
            model.append(nn.GELU())
        self.preprocess = nn.Sequential(*model)
    def forward(self, x):
        return self.preprocess(x)

class hsi_unet_model_gelu_pca(nn.Module):
    def __init__(self, in_channels, out_channels=10, features=[64, 128, 256, 512]):
        super(hsi_unet_model_gelu_pca,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block_gelu(in_channels,features[0])
        self.conv2 = encoding_block_gelu(features[0],features[1])
        self.conv3 = encoding_block_gelu(features[1],features[2])
        self.conv4 = encoding_block_gelu(features[2],features[3])
        self.conv5 = encoding_block_gelu(features[3]*2,features[3])
        self.conv6 = encoding_block_gelu(features[3],features[2])
        self.conv7 = encoding_block_gelu(features[2],features[1])
        self.conv8 = encoding_block_gelu(features[1],features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block_gelu(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

class hsi_unet_model_gelu(nn.Module):
    def __init__(self, in_channels, out_channels=10, features=[64, 128, 256, 512]):
        super(hsi_unet_model_gelu,self).__init__()
        self.preprocess = preprocessing_block(in_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block_gelu(in_channels,features[0])
        self.conv2 = encoding_block_gelu(features[0],features[1])
        self.conv3 = encoding_block_gelu(features[1],features[2])
        self.conv4 = encoding_block_gelu(features[2],features[3])
        self.conv5 = encoding_block_gelu(features[3]*2,features[3])
        self.conv6 = encoding_block_gelu(features[3],features[2])
        self.conv7 = encoding_block_gelu(features[2],features[1])
        self.conv8 = encoding_block_gelu(features[1],features[0])
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)
        self.bottleneck = encoding_block_gelu(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        x = self.preprocess(x)
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

###############################################################
############### HSI/RGB UNET Feature Fusion ###################
###############################################################

class unet_model_gelu_sensorfusion(nn.Module):
    def __init__(self,in_channels_hsi, out_channels=10):
        super(unet_model_gelu_sensorfusion,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1_rgb = encoding_block_gelu_2_conv_init_hsi(3, 64)
        self.conv1_hsi = encoding_block_gelu_2_conv_init_hsi(in_channels_hsi, 64)
        self.conv2 = encoding_block_gelu_2_conv(64, 128)
        self.conv3 = encoding_block_gelu_2_conv(128, 256)
        self.conv4 = encoding_block_gelu_2_conv(256, 512)
        self.conv_bridge = encoding_block_gelu_2_conv(512, 1024)
        self.deconv_bridge_1 = encoding_block_gelu_2_conv(2048, 1024)
        self.deconv_bridge_2 = encoding_block_gelu_2_conv(1024, 512)
        self.deconv4 = encoding_block_gelu_3_conv(1536, 1024, 512)
        self.deconv3 = encoding_block_gelu_3_conv(768, 512, 256)
        self.deconv2 = encoding_block_gelu_3_conv(384, 256, 128)
        self.deconv1 = encoding_block_gelu_3_conv(192, 128, 64)
        self.tconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_layer = nn.Conv2d(64,out_channels,kernel_size=1)
    def forward(self, x_rgb, x_hsi):
        skip_connections_rgb = []
        skip_connections_hsi = []
        
        #rgb downsampling
        x_rgb = self.conv1_rgb(x_rgb) # 320, 320, 3 -> 320, 320, 64
        skip_connections_rgb.append(x_rgb)
        x_rgb = self.pool(x_rgb)
        
        x_rgb = self.conv2(x_rgb) # 320, 320, 64 -> 160, 160, 128 
        skip_connections_rgb.append(x_rgb)
        x_rgb = self.pool(x_rgb)
        
        x_rgb = self.conv3(x_rgb) # 160, 160, 128 -> 80, 80, 256
        skip_connections_rgb.append(x_rgb)
        x_rgb = self.pool(x_rgb)
        
        x_rgb = self.conv4(x_rgb) # 80, 80, 256 -> 40, 40, 512
        skip_connections_rgb.append(x_rgb)
        x_rgb = self.pool(x_rgb)
        
        x_rgb = self.conv_bridge(x_rgb) # 40, 40, 512 -> 20, 20, 1024
        skip_connections_rgb = skip_connections_rgb[::-1] #reverses order of list
        
        #hsi downsampling
        x_hsi = self.conv1_hsi(x_hsi) # 320, 320, 3 -> 320, 320, 64
        skip_connections_hsi.append(x_hsi)
        x_hsi = self.pool(x_hsi)
        
        x_hsi = self.conv2(x_hsi) # 320, 320, 64 -> 
        skip_connections_hsi.append(x_hsi)
        x_hsi = self.pool(x_hsi)
        
        x_hsi = self.conv3(x_hsi) # 160, 160, 128 -> 80, 80, 256
        skip_connections_hsi.append(x_hsi)
        x_hsi = self.pool(x_hsi)
        
        x_hsi = self.conv4(x_hsi) # 80, 80, 256 -> 40, 40, 512
        skip_connections_hsi.append(x_hsi)
        x_hsi = self.pool(x_hsi)
        
        x_hsi = self.conv_bridge(x_hsi) # 40, 40, 512 -> 20, 20, 1024
        skip_connections_hsi = skip_connections_hsi[::-1] #reverses order of list
        
        #bridge
        x_comb = torch.cat((x_rgb, x_hsi), dim=1) #-> 20, 20, 2048
        x_comb = self.deconv_bridge_1(x_comb) # 20, 20, 2048 -> 20, 20, 1024
        
        # combined upsampling
        x_comb = self.tconv5(x_comb) # 20, 20, 1024 -> 40, 40, 512
        #x_comb = self.deconv_bridge_2(x_comb) # 40, 40, 1024 -> 40, 40, 512
        x_comb = torch.cat((skip_connections_rgb[0], skip_connections_hsi[0], x_comb), dim=1) #-> 40, 40, 1536
        print(x_comb.shape)
        
        x_comb = self.deconv4(x_comb) # 40, 40, 1536 -> 40, 40, 512
        x_comb = self.tconv4(x_comb) # 40, 40, 512 -> 80, 80, 512
        x_comb = torch.cat((skip_connections_rgb[1], skip_connections_hsi[1], x_comb), dim=1) #-> 80, 80, 768
        
        x_comb = self.deconv3(x_comb) # 80, 80, 768 -> 80, 80, 256
        x_comb = self.tconv3(x_comb) # 80, 80, 256 -> 160, 160, 128
        x_comb = torch.cat((skip_connections_rgb[2], skip_connections_hsi[2], x_comb), dim=1) #-> 160, 160, 384
        
        x_comb = self.deconv2(x_comb) # 160, 160, 384 -> 160, 160, 128
        x_comb = self.tconv2(x_comb) # 160, 160, 128 -> 320, 320, 64
        x_comb = torch.cat((skip_connections_rgb[3], skip_connections_hsi[3], x_comb), dim=1) #-> 320, 320, 192
        
        x_comb = self.deconv1(x_comb) # 320, 320, 192 -> 320, 320, 64
        x_comb = self.final_layer(x_comb)
        
        return x_comb
