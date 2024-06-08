import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import random

#augmentation
from albumentations.pytorch import ToTensorV2
import albumentations as A

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
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import skimage.io as skio # lighter dependency than tensorflow for working with our tensors/arrays

###########################################################
#################### Augmentations ########################
###########################################################

transformation = A.Compose([
    A.Resize(640,640),
    A.RandomCrop(width=320, height=320),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RGBShift(r_shift_limit=(0,0.1), g_shift_limit=0, b_shift_limit=0, p=0.5),
    #A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #A.augmentations.transforms.Normalize(mean=img_mean, std=img_std),
    ToTensorV2()
])

transformation_resize_img=A.Compose([
    A.Resize(960,960),
    #ToTensorV2()
])

transformation_inference=A.Compose([
    A.Resize(320, 320),
    ToTensorV2()
])

test_transformation = A.Compose([
    A.Resize(320,320),
    #A.augmentations.transforms.Normalize(mean=img_mean, std=img_std),
    #A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

hsi_mask_crop = A.Compose([
    A.Crop(x_min=0, y_min=0, x_max=320, y_max=672)
])

hsi_transformation = A.Compose([
    #A.Resize(640,640),
    A.RandomCrop(width=224, height=224),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    #A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.25), contrast_limit=(-0.1, 0.15), brightness_by_max=False),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.RGBShift(r_shift_limit=(0,0.1), g_shift_limit=0, b_shift_limit=0, p=0.5),
    #A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #A.augmentations.transforms.Normalize(mean=img_mean, std=img_std),
    ToTensorV2()
])

test_hsi_transformation = A.Compose([
    #A.RGBShift(r_shift_limit=(0,0.1), g_shift_limit=0, b_shift_limit=0, p=0.5),
    #A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #A.augmentations.transforms.Normalize(mean=img_mean, std=img_std),
    ToTensorV2()
])

sf_transformation = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
],
additional_targets={'image1':'image'}
)

sf_no_transformation = A.Compose([
    ToTensorV2()
],
additional_targets={'image1':'image'}
)

###############################################################
############ Custom Dataset and Preprocessing  ################
###############################################################

#replace mask values with smaller numbers
def replace_np_values(np_array, defects_only=False):

    value_to_replace = -1
    new_value = 0
    np_array[np_array == value_to_replace] = new_value 
    
    value_to_replace = 10
    new_value = 0
    np_array[np_array == value_to_replace] = new_value 
    
    #value_to_replace = 1
    #new_value = 1
    #np_array[np_array == value_to_replace] = new_value    

    #value_to_replace = 2
    #new_value = 2
    #np_array[np_array == value_to_replace] = new_value
    
    value_to_replace = 4
    new_value = 3
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 8
    new_value = 4
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 16
    new_value = 5
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 32
    new_value = 6
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 64
    new_value = 7
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 128
    new_value = 8
    np_array[np_array == value_to_replace] = new_value
    
    value_to_replace = 256
    new_value = 9
    np_array[np_array == value_to_replace] = new_value

    value_to_replace = 512
    new_value = 9
    np_array[np_array == value_to_replace] = new_value
    
    value_to_replace = 512
    new_value = 9
    np_array[np_array == value_to_replace] = new_value

    if defects_only:
        value_to_replace = 1
        new_value = 0
        np_array[np_array == value_to_replace] = new_value

# finds out if image contains any defects
def img_contains_defects(mask):
    if (torch.any(mask == 2) or torch.any(mask == 3) or torch.any(mask == 4) or torch.any(mask == 5) or torch.any(mask == 6) or torch.any(mask == 7) or torch.any(mask == 9)):
        return True
    else:
        return False

def img_contains_nothing(mask):
    if torch.all(mask == 0):
        return False
    else:
        return True


################################ RGB #####################################
class _WHDataset_10_classes(Dataset):
    def __init__(self, img_dir, mask_dir, transform_resize_img_only=None, transform=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(img_dir)
        self.transform_resize_img_only=transform_resize_img_only
        
    def __len__(self):
        return len(self.images)
    
    def get_image_mask_name(self, idx):
        mask_name = os.path.join(self.mask_dir, self.images[idx].replace('.png', '.npy'))
        return os.path.basename(mask_name)
    
    def __getitem__(self, idx):
        img_name=os.path.join(self.img_dir, self.images[idx])
        mask_name=os.path.join(self.mask_dir, self.images[idx].replace('.png', '.npy'))
        
        #mask_name=os.path.join(self.mask_dir, self.images[idx].replace('.png', '.npy'))  #masks are loaded up as npy files
        
        #read in image as PIL and mask as numpy
        image=np.array(Image.open(img_name).convert('RGB'))

        mask=np.load(mask_name)

        image=image/255

        if self.transform_resize_img_only:
            transformed=self.transform_resize_img_only(image=image)
            image = transformed["image"]
        
        #loops around to find transformed images with defects, after 7 loops it just takes whatever it finds
        if self.transform:
            for i in range(7):
                transformed = self.transform(image=image, mask=mask)
                image_trans = transformed["image"]
                mask_trans = transformed["mask"]
                #mask_trans=mask_trans[:,:,0] --> masks are not 3d
                if img_contains_defects(mask_trans):
                    break;
                if img_contains_nothing(mask_trans):
                    i = i - 1

        return image_trans, mask_trans

################################ HSI #####################################

class _WH_HSI_Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, hsi_mask_crop=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.hsi_mask_crop = hsi_mask_crop
        self.images=os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name=os.path.join(self.img_dir, self.images[idx])

        mask_name=os.path.join(self.mask_dir, self.images[idx])

        #read in image as PIL and mask as numpy

        image=np.load(img_name)
        #sliced_image = image.copy()[:,:,::4]
        mask=np.load(mask_name)

        #crop hsi mask into same height and width as images
        if self.hsi_mask_crop:
          cropped = self.hsi_mask_crop(image=mask)
          mask_cropped = cropped["image"]

        #replace mask values with 0,1,2,3,4,5, etc.
        replace_np_values(mask_cropped, defects_only=False)

        # normalize image and convert to float32
        image=(image/4096).astype(np.float32)
        #print(image.dtype)

        #loops around to find transformed images with defects, after 7 loops it just takes whatever it finds
        if self.transform:
            for i in range(14):
                transformed = self.transform(image=image, mask=mask_cropped)
                image_trans = transformed["image"]
                mask_trans = transformed["mask"]
                #mask_trans=mask_trans[:,:,0]
                if img_contains_defects(mask_trans):
                    break;
                if img_contains_nothing(mask_trans):
                    i = i - 1

        return image_trans, mask_trans

################################ Sensor Fusion #####################################

class _WH_RGB_HSI_Dataset(Dataset):
    def __init__(self, rgb_img_dir, hsi_img_dir, mask_dir, transform=None):
        self.rgb_img_dir=rgb_img_dir
        self.hsi_img_dir=hsi_img_dir
        self.mask_dir=mask_dir
        self.transform=transform

        self.rgb_images=os.listdir(rgb_img_dir)
        self.hsi_images=os.listdir(hsi_img_dir)
        
    def __len__(self):
        return len(self.hsi_images)

    def __getitem__(self, idx):
        
        rgb_img_name=os.path.join(self.rgb_img_dir, self.rgb_images[idx])
        hsi_img_name=os.path.join(self.hsi_img_dir, self.rgb_images[idx].replace('.png', '.npy'))
        mask_name=os.path.join(self.mask_dir, self.rgb_images[idx].replace('.png', '.npy'))

        #read in RGB image as PIL
        rgb_image=np.array(Image.open(rgb_img_name).convert('RGB'))/255

        #read in HSI image and mask as numpy
        hsi_image=np.load(hsi_img_name).astype(np.float32) #care how many channels the HSI images have
        mask = np.load(mask_name)
        
        #replace mask values with 0,1,2,3,4,5, etc.
        replace_np_values(mask, defects_only=False)


        #loops around to find transformed images with defects, after 7 loops it just takes whatever it finds
        if self.transform:
            for i in range(14):
                transformed = self.transform(image=rgb_image, image1 = hsi_image, mask=mask)
                rgb_image_trans = transformed["image"]
                hsi_image_trans = transformed["image1"]
                mask_trans = transformed["mask"]

                #check if mask onctains defects, if not then reroll
                if img_contains_defects(mask_trans):
                    break;
                if img_contains_nothing(mask_trans):
                    i = i - 1

        return rgb_image_trans, hsi_image_trans, mask_trans

#################################################
################  Constants  ####################
#################################################

print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_today=datetime.today().strftime('%Y-%m-%d')

#color values for complete classes
NUM_UNIQUE_VALUES_LONG = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
COLORS_LONG = ['black',          'white', 'green',  'red',  'cyan',          'blue',      'darkred',     'pink',     'navy', 'orange', ]
CLASSES_LONG = ['background', 'chicken_front', 'chicken_back', 'Blood', 'Bones', 'SurfaceDefect', 'Discoloring', 'Scalding', 'Deformed', 'Fat/Skin']
CLASSES_LONG_9 = ['background', 'chicken_front', 'chicken_back', 'Blood', 'Bones', 'SurfaceDefect', 'Discoloring', 'Scalding', 'Fat/Skin']
TXT_COLORS_LONG=['\033[0mblack', '\033[94mwhite', '\033[32mgreen','\033[91mred', 
                 '\033[96mcyan', '\033[94mblue', '\\033[31mdarkred',
                 '\033[95mpink' ,'\033[34mnavy' , '\033[38;2;255;165;0morange']
TXT_COLORS_LONG_COLOR_ONLY=['\033[0m', '\033[94m', '\033[32m', '\033[91m', '\033[96m', '\033[94m', '\033[31m', '\033[95m' ,'\033[34m' , '\033[38;2;255;165;0m']

cmap_long = ListedColormap(COLORS_LONG)
BOUNDARIES_LONG = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
norm_long = BoundaryNorm(BOUNDARIES_LONG, len(COLORS_LONG))

N_CLASSES = 10

HSI_HEIGHT = 672
HSI_WIDTH = 320

##########################################################
################  Training Functions  ####################
##########################################################

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

###################################################################################

        ###############################################################
        ############### sensor fusion model training ##################
        ###############################################################

###################################################################################

# sensor fusion model training with two possible loss functions
def sf_model_training_multiloss(model, train_loader, val_loader, num_epochs, ce_loss_fn, dice_loss_fn, optimizer, scaler, scheduler, 
                            avg_train_loss_list, avg_val_loss_list, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE,
                             activate_scheduler=True, patience=15, model_name=''):

    _today=datetime.today().strftime('%Y-%m-%d')
    print('Training beginning with following parameters:')
    print(f'No. Epochs: {num_epochs}')

    #this is used for early stopping, value should be pretty large
    best_val_loss = 1000
    
    #training with Cross Entropy Loss
    for epoch in range(num_epochs):
        
        print(f'Epoch: {epoch}')
        train_batch_loss=0
        val_batch_loss=0
        train_batch_iou=0
        val_batch_iou=0
        
        #####################################################
        ############### training instance ###################
        #####################################################
        
        model.train()
        train_loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (rgb_img, hsi_img, mask) in train_loop:
            
            rgb_img = rgb_img.to(DEVICE)
            hsi_img = hsi_img.to(DEVICE)
            mask = mask.to(DEVICE)
            mask = mask.type(torch.long)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(rgb_img.float(), hsi_img)
                ce_loss = ce_loss_fn(predictions, mask)
                dice_loss = dice_loss_fn(predictions, mask)
                loss = ce_loss + dice_loss
                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # update tqdm loop
            train_loop.set_postfix(loss=loss.item())
            
            train_batch_loss = train_batch_loss + loss.item()
            '''
            for k in range(TRAIN_BATCH_SIZE):
            
                #calculate batch iou
                pred_combined_mask=process_prediction_to_combined_mask(predictions)
                
                #batch iou
                train_batch_iou=train_batch_iou+calculate_img_iou(iou_all_classes(pred_combined_mask[k,:,:], mask[k,:,:]))
            '''
    
        #calculate average loss
        print(f'Average Train Batch Loss: {train_batch_loss/TRAIN_BATCH_SIZE:.4f}')
        avg_train_loss_list.append(train_batch_loss/TRAIN_BATCH_SIZE)
        
        ####################################################
        ############## validation instance #################
        ####################################################
        
        model.eval()
        val_loop = tqdm(enumerate(val_loader),total=len(val_loader))
        for batch_idx, (rgb_img, hsi_img, mask) in val_loop:
            
            with torch.no_grad():
                
                rgb_img = rgb_img.to(DEVICE)
                hsi_img = hsi_img.to(DEVICE)
                mask = mask.to(DEVICE)
                mask = mask.type(torch.long)
            
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(rgb_img.float(), hsi_img)
                    ce_loss = ce_loss_fn(predictions, mask)
                    dice_loss = dice_loss_fn(predictions, mask)
                    val_loss = ce_loss + dice_loss

    
            # update tqdm loop
            val_loop.set_postfix(val_loss=val_loss.item())
            
            val_batch_loss = val_batch_loss + val_loss.item()

        avg_val_loss = val_batch_loss / len(val_loader)
        print(f'Average Validation Batch Loss: {val_batch_loss/VAL_BATCH_SIZE:.4f}')        
        avg_val_loss_list.append(val_batch_loss/VAL_BATCH_SIZE)
        
        #######################################################
        ############### adjust learning rate ##################
        #######################################################
        
        if activate_scheduler:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Adam lr {before_lr:.4f} -> {after_lr:.4f}")
        
        ###################################################################################################################
        ############## visualize training and validation results and also save model after 50 epochs ######################
        ###################################################################################################################
            
        if ((epoch%10==0) and (epoch>0) or (epoch==num_epochs)):
            
            plot_range=range(epoch)
            
            fig, axs = plt.subplots(figsize=(9,6))
            
            ###
            # Loss
            ###
            
            axs.plot(range(len(avg_train_loss_list)), avg_train_loss_list, marker='o', linestyle='-', label='Training Loss', color='blue')
            
            #create twin axis
            ax2 = axs.twinx()
            ax2.plot(range(len(avg_val_loss_list)), avg_val_loss_list, marker='o', linestyle='-', label='Validation Loss', color='orange')
            
            # Add labels and title
            axs.set_xlabel('Epochs')
            axs.set_ylabel('Training Loss', color='blue')
            ax2.set_ylabel('Validation Loss', color='orange')
            axs.set_title('Training vs Validation Loss')
            
            # Show legend for both axes
            axs.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
            plt.grid(True)
            plt.show()
            
        ######################################################
        ################# early stopping #####################
        ######################################################
        if patience > 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    # Save the best model
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scaler_state_dict': scaler.state_dict()
                    }, f'best_model_{_today}_{model_name}.pt')
                    return best_model, loss, avg_train_loss_list, avg_val_loss_list
                    break

        if epoch==50 or epoch==75 or epoch==(num_epochs-1):
            # Save all the elements to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scaler_state_dict': scaler.state_dict()
            }, f'model_e{epoch}_{_today}_{model_name}.pt')
            
    return model, loss, avg_train_loss_list, avg_val_loss_list


###################################################################################

        ###############################################################
        ################### regular model training ####################
        ###############################################################

###################################################################################

# model training with two possible loss functions
def model_training_multiloss(model, train_loader, val_loader, num_epochs, ce_loss_fn, dice_loss_fn, optimizer, scaler, scheduler, 
                            avg_train_loss_list, avg_val_loss_list, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE,
                             activate_scheduler=True, patience=15, model_name=''):

    #this is used for early stopping, value should be pretty large
    best_val_loss = 1000
    _today=datetime.today().strftime('%Y-%m-%d')
    print('Training beginning with following parameters:')
    print(f'No. Epochs: {num_epochs}')
    
    #training with Cross Entropy Loss
    for epoch in range(num_epochs):
        
        print(f'Epoch: {epoch}')
        train_batch_loss=0
        val_batch_loss=0
        train_batch_iou=0
        val_batch_iou=0
        
        #####################################################
        ############### training instance ###################
        #####################################################
        
        model.train()
        train_loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (img, _, mask) in train_loop:
            
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            mask = mask.type(torch.long)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(img.float())
                ce_loss = ce_loss_fn(predictions, mask)
                dice_loss = dice_loss_fn(predictions, mask)
                loss = ce_loss + dice_loss
                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # update tqdm loop
            train_loop.set_postfix(loss=loss.item())
            
            train_batch_loss = train_batch_loss + loss.item()
    
        #calculate average loss
        print(f'Average Train Batch Loss: {train_batch_loss/TRAIN_BATCH_SIZE:.4f}')
        avg_train_loss_list.append(train_batch_loss/TRAIN_BATCH_SIZE)
        
        ####################################################
        ############## validation instance #################
        ####################################################
        
        model.eval()
        val_loop = tqdm(enumerate(val_loader),total=len(val_loader))
        for batch_idx, (img, _, mask) in val_loop:
            
            with torch.no_grad():
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                mask = mask.type(torch.long)
            
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(img.float())
                    ce_loss = ce_loss_fn(predictions, mask)
                    dice_loss = dice_loss_fn(predictions, mask)
                    val_loss = ce_loss + dice_loss

    
            # update tqdm loop
            val_loop.set_postfix(val_loss=val_loss.item())
            
            val_batch_loss = val_batch_loss + val_loss.item()

        avg_val_loss = val_batch_loss / len(val_loader)
        print(f'Average Validation Batch Loss: {val_batch_loss/VAL_BATCH_SIZE:.4f}')        
        avg_val_loss_list.append(val_batch_loss/VAL_BATCH_SIZE)
        
        #######################################################
        ############### adjust learning rate ##################
        #######################################################
        
        if activate_scheduler:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Adam lr {before_lr:.4f} -> {after_lr:.4f}")
        
        ###################################################################################################################
        ############## visualize training and validation results and also save model after 50 epochs ######################
        ###################################################################################################################
            
        if ((epoch%10==0) and (epoch>0) or (epoch==num_epochs)):
            
            plot_range=range(epoch)
            
            fig, axs = plt.subplots(figsize=(9,6))
            
            ###
            # Loss
            ###
            
            axs.plot(range(len(avg_train_loss_list)), avg_train_loss_list, marker='o', linestyle='-', label='Training Loss', color='blue')
            
            #create twin axis
            ax2 = axs.twinx()
            ax2.plot(range(len(avg_val_loss_list)), avg_val_loss_list, marker='o', linestyle='-', label='Validation Loss', color='orange')
            
            # Add labels and title
            axs.set_xlabel('Epochs')
            axs.set_ylabel('Training Loss', color='blue')
            ax2.set_ylabel('Validation Loss', color='orange')
            axs.set_title('Training vs Validation Loss')
            
            # Show legend for both axes
            axs.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
            plt.grid(True)
            plt.show()
            
        ######################################################
        ################# early stopping #####################
        ######################################################

        if patience > 0 and epoch>int(1/3*num_epochs):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scaler_state_dict': scaler.state_dict()
                }, f'best_model_{_today}_{model_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                
        if epoch==50 or epoch==75 or epoch==(num_epochs-1):
            # Save all the elements to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scaler_state_dict': scaler.state_dict()
            }, f'model_e{epoch}_{_today}_{model_name}.pt')
            
    return model, loss, avg_train_loss_list, avg_val_loss_list

###################################################################################

        ###############################################################
        ################### regular model training ####################
        ###############################################################

###################################################################################

# model training with one loss function
def model_training(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, scaler, scheduler, 
                            avg_train_loss_list, avg_val_loss_list, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE,
                            activate_scheduler=True, patience=15, model_name=''):

    #this is used for early stopping, value should be pretty large
    best_val_loss = 1000
    _today=datetime.today().strftime('%Y-%m-%d')
    print('Training beginning with following parameters:')
    print(f'No. Epochs: {num_epochs}')
    
    #training with Cross Entropy Loss
    for epoch in range(num_epochs):
        
        print(f'Epoch: {epoch}')
        train_batch_loss=0
        val_batch_loss=0
        train_batch_iou=0
        val_batch_iou=0
        
        #####################################################
        ############### training instance ###################
        #####################################################
        
        model.train()
        train_loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for batch_idx, (img, _, mask) in train_loop:
            
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            mask = mask.type(torch.long)
            
            # forward
            with torch.cuda.amp.autocast():
                predictions = model(img.float())
                loss = loss_fn(predictions, mask)

                
            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # update tqdm loop
            train_loop.set_postfix(loss=loss.item())
            
            train_batch_loss = train_batch_loss + loss.item()
    
        #calculate average loss
        print(f'Average Train Batch Loss: {train_batch_loss/TRAIN_BATCH_SIZE:.4f}')
        avg_train_loss_list.append(train_batch_loss/TRAIN_BATCH_SIZE)
        
        ####################################################
        ############## validation instance #################
        ####################################################
        
        model.eval()
        val_loop = tqdm(enumerate(val_loader),total=len(val_loader))
        for batch_idx, (img, _, mask) in val_loop:
            
            with torch.no_grad():
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                mask = mask.type(torch.long)
            
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(img.float())
                    val_loss = loss_fn(predictions, mask)

    
            # update tqdm loop
            val_loop.set_postfix(val_loss=val_loss.item())
            
            val_batch_loss = val_batch_loss + val_loss.item()

        avg_val_loss = val_batch_loss / len(val_loader)
        print(f'Average Validation Batch Loss: {val_batch_loss/VAL_BATCH_SIZE:.4f}')        
        avg_val_loss_list.append(val_batch_loss/VAL_BATCH_SIZE)
        
        #######################################################
        ############### adjust learning rate ##################
        #######################################################
        
        if activate_scheduler:
            before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Adam lr {before_lr:.4f} -> {after_lr:.4f}")
        
        ###################################################################################################################
        ############## visualize training and validation results and also save model after 50 epochs ######################
        ###################################################################################################################
            
        if ((epoch%10==0) and (epoch>0) or (epoch==num_epochs)):
            
            plot_range=range(epoch)
            
            fig, axs = plt.subplots(figsize=(9,6))
            
            ###
            # Loss
            ###
            
            axs.plot(range(len(avg_train_loss_list)), avg_train_loss_list, marker='o', linestyle='-', label='Training Loss', color='blue')
            
            #create twin axis
            ax2 = axs.twinx()
            ax2.plot(range(len(avg_val_loss_list)), avg_val_loss_list, marker='o', linestyle='-', label='Validation Loss', color='orange')
            
            # Add labels and title
            axs.set_xlabel('Epochs')
            axs.set_ylabel('Training Loss', color='blue')
            ax2.set_ylabel('Validation Loss', color='orange')
            axs.set_title('Training vs Validation Loss')
            
            # Show legend for both axes
            axs.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
            plt.grid(True)
            plt.show()
            
        ######################################################
        ################# early stopping #####################
        ######################################################

        if patience > 0 and epoch>int(1/3*num_epochs):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scaler_state_dict': scaler.state_dict()
                }, f'best_model_{_today}_{model_name}.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
                
        if epoch==50 or epoch==75 or epoch==(num_epochs-1):
            # Save all the elements to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scaler_state_dict': scaler.state_dict()
            }, f'model_e{epoch}_{_today}_{model_name}.pt')
            
    return model, loss, avg_train_loss_list, avg_val_loss_list

def load_model(model_type, optimizer, scaler, model_path):
    model = model_type.to(DEVICE)
    checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return model

#makes training etc. deterministic
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

