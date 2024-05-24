import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime

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
from torchsummary import summary

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
import skimage.io as skio # lighter dependency than tensorflow for working with our tensors/arrays

from thesis_constants.functions_and_constants import *

###############################################
############### Visualisation ####################
###############################################

def rgb_visualize_prediction_vs_ground_truth_single_batches_before_argmax(model, loader, height, width):
    
    model.eval()
    
    #for checking if masks fit to respective image, all defects are displayed in a unique color
    
    print('Legend:')
    for i, color in enumerate(COLORS_LONG):
        print(f'{TXT_COLORS_LONG[i]} -> {CLASSES_LONG[i]}')
        
    print('\033[0m- - - - - - -')
    
    batch=next(iter(loader))

    img, mask=batch
    
    img = img.to(DEVICE)
    mask = mask.to(DEVICE)
    
    softmax = nn.Softmax(dim=1)
    prob_pred_mask = softmax(model(img.float())).to('cpu')
    pred_mask = torch.argmax(softmax(model(img.float())),axis=1).to('cpu')
    #prob_pred_mask = softmax(model(img)).detach.to('cpu').numpy()
    
    #print(np.max(prob_pred_mask[0,c,:,:]))

    #loop over all images inside the batch
    for j in range(img.shape[0]):
        
        #list for storing the masks
        prob_masks=[]
        
        #loop over all class masks in probability mask
        for c in range(len(NUM_UNIQUE_VALUES_LONG)):
            #normalize probability of predicted masks
            #prob_pred_mask[j,c,:,:]=prob_pred_mask[j,c,:,:]/torch.max(prob_pred_mask[j,c,:,:])
            prob_masks.append(prob_pred_mask[j,c,:,:])
        
        # Initialize an empty overlay
        overlay = np.zeros((height, width, 3))  
        for i, prob_mask in enumerate(prob_masks):

            color = np.array(plt.cm.colors.to_rgba(COLORS_LONG[i])[:3])  # Get color for class
            overlay += np.dstack((color[0] * prob_mask.detach().numpy(), 
                                  color[1] * prob_mask.detach().numpy(), 
                                  color[2] * prob_mask.detach().numpy()))  # Add color with transparency

        # Clip overlay to ensure values are between 0 and 1
        overlay = np.clip(overlay, 0, 1)
            
        fig , axs =  plt.subplots(1, 4, figsize=(24, 24))
    
        print(f'Image No.{j}')
        #convert into arrays for visualisation
        single_img = np.asarray(img[j,:,:,:].to('cpu').permute(1,2,0))
        single_mask = np.asarray(mask[j,:,:].to('cpu'))
        single_pred = np.asarray(pred_mask[j,:,:].to('cpu'))

        axs[0].set_title('Image')
        axs[1].set_title('Ground Truth')
        axs[2].set_title('Prediction')
        axs[3].set_title('Prediction Probabilities')
        axs[0].imshow(single_img)
        axs[1].imshow(single_mask, cmap=cmap_long, norm=norm_long)
        axs[2].imshow(single_pred, cmap=cmap_long, norm=norm_long)
        axs[3].imshow(overlay)
        
    fig.show()
        
    return pred_mask

def rgb_visualize_prediction_vs_ground_truth_single_images_overlay(img, truth_mask, pred_mask, is_img_normalized=False):

    if is_img_normalized:

        fig , axs =  plt.subplots(1, 3, figsize=(18, 12))

        denorm_img = img.numpy().transpose((1, 2, 0)).copy()

        axs[0].set_title('Normalized Image')
        axs[1].set_title('Denormalized Image')
        axs[2].set_title('Image with Ground Truth')
        axs[3].set_title('Image with Prediction')

        axs[0].imshow(np.asarray(img.to('cpu').permute(1,2,0)))
        axs[1].imshow(np.asarray(denorm_img.to('cpu').permute(1,2,0)))
        axs[2].imshow(np.asarray(img.to('cpu').permute(1,2,0)))
        axs[3].imshow(np.asarray(img.to('cpu').permute(1,2,0)))

        axs[2].imshow(np.asarray(truth_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[3].imshow(np.asarray(pred_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)

    else:

        fig , axs =  plt.subplots(1, 3, figsize=(18, 12))

        axs[0].set_title('Plain Image')
        axs[1].set_title('Image with Ground Truth')
        axs[2].set_title('Image with Prediction')

        axs[0].imshow(np.asarray(img.to('cpu').permute(1,2,0)))
        axs[1].imshow(np.asarray(img.to('cpu').permute(1,2,0)))
        axs[2].imshow(np.asarray(img.to('cpu').permute(1,2,0)))

        axs[1].imshow(np.asarray(truth_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[2].imshow(np.asarray(pred_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)

    plt.show()

def rgb_visualize_prediction_vs_ground_truth_single_images_overlay_postprocessed(img, truth_mask, pred_mask, processed_pred_mask, is_img_normalized=False):

    fig , axs =  plt.subplots(2, 2, figsize=(16, 12))

    axs[0, 0].set_title('Plain Image')
    axs[0, 1].set_title('Image with Ground Truth')
    axs[1, 0].set_title('Image with Prediction')
    axs[1, 1].set_title('Image with Post Processing')

    axs[0, 0].imshow(img)
    axs[0, 1].imshow(img)
    axs[1, 0].imshow(img)
    axs[1, 1].imshow(img)

    axs[0, 1].imshow(truth_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)
    axs[1, 0].imshow(pred_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)
    axs[1, 1].imshow(processed_pred_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')

    plt.show()

def iou_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, print_iou=False, SINGLE_PREDICTION=False):
    
    iou_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        # condition for ground truth mask being all 0s    
        if one_hot_truth_masks[:,:,i].eq(0).all() and one_hot_pred_masks[:,:,i].eq(0).all():
            iou_list.append(-1.0)
            if print_iou:
                print(f'Prediction Mask {i} is empty')
            
        
        # condition for prediction mask being all 0s
        #if one_hot_pred_masks[:,:,i].eq(0).all():
        #    print(f'Prediction Mask {i} is empty')
        #    iou_list.append(-1)
        
        else:
            if SINGLE_PREDICTION:
                union=one_hot_pred_masks.squeeze(0)[:,:,i]|one_hot_truth_masks[:,:,i]
                intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
            else:
                union=one_hot_pred_masks[:,:,i]|one_hot_truth_masks[:,:,i]
                intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
            
            iou=intersection.sum().item()/(union.sum().item()+1e-8)
            
            if print_iou:
                print(f'Prediction Mask {i} has IOU of {iou}')
            
            iou_list.append(iou)
            
    return iou_list

def calculate_img_iou(iou_array, N_CLASS=N_CLASSES, IGNORE_N_CLASSES=3):
    
    iou=0
    n_classes=IGNORE_N_CLASSES
    
    for i in range(N_CLASS):
        #skip background and chicken filet iou
        if i >= IGNORE_N_CLASSES:
            #negative numbers mean that masks in ground truth were empty
            if iou_array[i] >= 0:

                iou=iou+iou_array[i]
                
            else:
                n_classes=n_classes+1

    class_iou = iou/(N_CLASS-n_classes+1e-8)
    
    return class_iou

def intersection_and_union_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, SINGLE_PREDICTION=False):
    
    intersection_list=[]
    union_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        if SINGLE_PREDICTION:
            union=one_hot_pred_masks.squeeze(0)[:,:,i]|one_hot_truth_masks[:,:,i]
            intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
        else:
            union=one_hot_pred_masks[:,:,i]|one_hot_truth_masks[:,:,i]
            intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
        
        intersection_list.append(intersection.sum().item())
        union_list.append(union.sum().item())
             
    return intersection_list, union_list

def dice_values_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, print_dice=False, SINGLE_PREDICTION=False):
    
    numinator_list=[]
    denominator_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        if SINGLE_PREDICTION:
            intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
            dice_numinator=2*intersection.sum().item()
            dice_denominator=one_hot_pred_masks.squeeze(0)[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
        else:
            intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
            dice_numinator=2*intersection.sum().item()
            dice_denominator=one_hot_pred_masks[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
        
        numinator_list.append(dice_numinator)
        denominator_list.append(dice_denominator)

    return numinator_list, denominator_list

#calcuates the dice score of a SINGLE prediction, not a single batch
def dice_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, print_dice=False, SINGLE_PREDICTION=False):
    
    dice_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        # condition for ground truth mask being all 0s    
        if one_hot_truth_masks[:,:,i].eq(0).all() and one_hot_pred_masks[:,:,i].eq(0).all():
            if print_dice:
                 print(f'Prediction Mask {i} is empty')
            dice_list.append(-1)
            
        else:
            if SINGLE_PREDICTION:
                intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
                dice_numinator=2*intersection.sum().item()
                dice_denominator=one_hot_pred_masks.squeeze(0)[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
            else:
                intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
                dice_numinator=2*intersection.sum().item()
                dice_denominator=one_hot_pred_masks[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
            
            dice=dice_numinator/(dice_denominator+1e-8)
            
            if print_dice:
                print(f'Prediction Mask {i} has Dice Score of {dice}')
                
            dice_list.append(dice)
            
    return dice_list

#basically same function as calculate_class_iou
def calculate_img_dice(iou_array, N_CLASS=N_CLASSES, IGNORE_N_CLASSES=3):
    
    iou=0
    n_classes=IGNORE_N_CLASSES
    
    for i in range(N_CLASS):
        #skip background and chicken filet iou
        if i >= IGNORE_N_CLASSES:
            
            #negative numbers mean that masks in ground truth were empty
            if iou_array[i] >= 0:

                iou=iou+iou_array[i]
                
            else:
                n_classes=n_classes+1

    class_iou = iou/(N_CLASS-n_classes+1e-8)
    
    return class_iou

def is_ground_truth_empty(truth_mask, N_CLASS=N_CLASSES):
    
    gt_array=[]

    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)

    for i in range(N_CLASS):
        if one_hot_truth_masks[:,:,i].eq(0).all():
            gt_array.append(True)
        else:
            gt_array.append(False)

    return gt_array

def is_prediction_empty(pred_mask, N_CLASS=N_CLASSES):
    
    pred_array=[]

    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)

    for i in range(N_CLASS):
        if one_hot_pred_masks[:,:,i].eq(0).all():
            pred_array.append(True)
        else:
            pred_array.append(False)

    return pred_array

# confusion matrix
def plot_confusion_matrix(gt_flat, pred_flat, label_array):
    conf=ConfusionMatrixDisplay.from_predictions(gt_flat, pred_flat, display_labels=label_array)

###############################################
############### Evaluation ####################
###############################################

def capture_model_metrics_pixelwise_and_confusion_matrix(model, test_dataset_final, visualize = True, 
                                                         confusion_matrix = True, norm_mode = 'pred',
                                                         smooth=1e-8):

    test_ds_union = [0,0,0,0,0,0,0,0,0,0]
    test_ds_intersection = [0,0,0,0,0,0,0,0,0,0]
    test_ds_numerator = [0,0,0,0,0,0,0,0,0,0]
    test_ds_denominator = [0,0,0,0,0,0,0,0,0,0]
    img_dice = []
    img_iou = []

    ground_truth_all_images=np.zeros((320,320, len(test_dataset_final)))
    prediction_all_images=np.zeros((320,320, len(test_dataset_final)))
    
    with torch.no_grad():
        model.eval()

        for n, (img, mask) in enumerate(test_dataset_final):

            # model prediction
            img = img.to(DEVICE).unsqueeze(0)
            mask = mask.to(DEVICE)

            softmax = nn.Softmax(dim=1)

            preds = torch.argmax(softmax(model(img.float())),axis=1).to('cpu').squeeze(0)

            # convert torch tensor to numpy array
            prediction_all_images[:,:,n] = preds.numpy()
            ground_truth_all_images[:,:,n] = mask.to('cpu').numpy()

            #calculate dice and iou score
            is_list, u_list=intersection_and_union_all_classes(mask, preds, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, preds, SINGLE_PREDICTION=True)

            if visualize:
                rgb_visualize_prediction_vs_ground_truth_single_images_overlay(img.squeeze(0), mask, preds.squeeze(0))

            print('IOU')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(preds)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {is_list[i]/(u_list[i]+smooth):.4f}')

                    #tracking class average of iou across all images
                    test_ds_union[i] += u_list[i]
                    test_ds_intersection[i] += is_list[i]
                    
            print(TXT_COLORS_LONG_COLOR_ONLY[0]+ 'x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x')

            print('Dice')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(preds)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {n_list[i]/(d_list[i]+smooth):.4f}')

                    #tracking class average of iou across all images
                    test_ds_numerator[i] += n_list[i]
                    test_ds_denominator[i] += d_list[i]
    
            print(TXT_COLORS_LONG_COLOR_ONLY[0])
    
            #only defects
            # calculate iou and dice

            intersection_array = np.array(is_list)
            union_array = np.array(u_list)
            numinator_array = np.array(n_list)
            denominator_array = np.array(d_list)

            iou_image_pixelwise = intersection_array[3:].sum()/(union_array[3:].sum()+smooth)
            dice_image_pixelwise = numinator_array[3:].sum()/(denominator_array[3:].sum()+smooth)

            print('Defects Only')
            print(TXT_COLORS_LONG_COLOR_ONLY[0]+f'Image Average IoU: {iou_image_pixelwise:.4f}')
            print(TXT_COLORS_LONG_COLOR_ONLY[0]+f'Image Average Dice: {dice_image_pixelwise:.4f}')

            img_dice.append(dice_image_pixelwise)
            img_iou.append(iou_image_pixelwise)

            # frees up memeory every 10th image
            if n%10:
                torch.cuda.empty_cache()

        if confusion_matrix:
            gt_flat = ground_truth_all_images.flatten()
            prediction_flat = prediction_all_images.flatten()

            fig, ax = plt.subplots(figsize=(10, 8))

            conf=ConfusionMatrixDisplay.from_predictions(gt_flat, prediction_flat, display_labels=CLASSES_LONG, normalize=norm_mode, ax=ax, xticks_rotation='vertical')

            plt.show()

            return test_ds_union, test_ds_intersection, test_ds_numerator, test_ds_denominator, iou_image_pixelwise, dice_image_pixelwise
        
def calculate_model_metrics(test_ds_union, 
                            test_ds_intersection, 
                            test_ds_numerator, 
                            test_ds_denominator):

    print('Average IoU over entire Test Dataset: '+f'{np.sum(np.array(test_ds_union))/(np.sum(np.array(test_ds_intersection))+1e-06):.4f}')
    print('Average Dice Score over entire Test Dataset: '+f'{np.sum(np.array(test_ds_numerator))/(np.sum(np.array(test_ds_denominator))+1e-06):.4f}')

    print('--Class Average IoU--')
    for i in range(9):

        print(f'{CLASSES_LONG[i]}: {test_ds_union[i]/(test_ds_intersection[i]+1e-06):.4f}')

    print('--Class Average Dice Score--')

    for i in range(9):

        print(f'{CLASSES_LONG[i]}: {test_ds_numerator[i]/(test_ds_denominator[i]+1e-06):.4f}')


def intersection_and_union_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, SINGLE_PREDICTION=False):
    
    intersection_list=[]
    union_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        if SINGLE_PREDICTION:
            union=one_hot_pred_masks.squeeze(0)[:,:,i]|one_hot_truth_masks[:,:,i]
            intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
        else:
            union=one_hot_pred_masks[:,:,i]|one_hot_truth_masks[:,:,i]
            intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
        
        intersection_list.append(intersection.sum().item())
        union_list.append(union.sum().item())
             
    return intersection_list, union_list

def dice_values_all_classes(truth_mask, pred_mask, N_CLASS=N_CLASSES, print_dice=False, SINGLE_PREDICTION=False):
    
    numinator_list=[]
    denominator_list=[]
    
    one_hot_pred_masks=F.one_hot(pred_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    one_hot_truth_masks=F.one_hot(truth_mask.to(torch.int64), num_classes=N_CLASS).to(DEVICE)
    
    for i in range(N_CLASS):
        
        if SINGLE_PREDICTION:
            intersection=one_hot_pred_masks.squeeze(0)[:,:,i]&one_hot_truth_masks[:,:,i]
            dice_numinator=2*intersection.sum().item()
            dice_denominator=one_hot_pred_masks.squeeze(0)[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
        else:
            intersection=one_hot_pred_masks[:,:,i]&one_hot_truth_masks[:,:,i]
            dice_numinator=2*intersection.sum().item()
            dice_denominator=one_hot_pred_masks[:,:,i].sum().item()+one_hot_truth_masks[:,:,i].sum().item()
        
        numinator_list.append(dice_numinator)
        denominator_list.append(dice_denominator)

    return numinator_list, denominator_list
