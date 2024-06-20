import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
import time

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
from thesis_constants.visualisation_and_evaluation import *

def capture_model_metrics_pixelwise_and_confusion_matrix_sf(model, test_dataset_final, data_source, visualize = True, 
                                                         confusion_matrix = True, norm_mode = 'pred', mask_shape=(320,320),
                                                         smooth=1e-8):

    test_ds_union = [0,0,0,0,0,0,0,0,0,0]
    test_ds_intersection = [0,0,0,0,0,0,0,0,0,0]
    test_ds_numerator = [0,0,0,0,0,0,0,0,0,0]
    test_ds_denominator = [0,0,0,0,0,0,0,0,0,0]
    img_dice = []
    img_iou = []

    ground_truth_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))
    prediction_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))
    
    with torch.no_grad():
        model.eval()

        for n, batch in enumerate(test_dataset_final):

            rgb_img, hsi_img, mask = batch

            # model prediction
            if data_source=='rgb':
                rgb_img = rgb_img.to(DEVICE).unsqueeze(0)
                mask = mask.to(DEVICE)

                softmax = nn.Softmax(dim=1)

                preds = torch.argmax(softmax(model(rgb_img.float())),axis=1).to('cpu').squeeze(0)

            if data_source=='hsi':
                hsi_img = hsi_img.to(DEVICE).unsqueeze(0)
                mask = mask.to(DEVICE)

                softmax = nn.Softmax(dim=1)

                preds = torch.argmax(softmax(model(hsi_img.float())),axis=1).to('cpu').squeeze(0)

            elif data_source=='sf':

                rgb_img = rgb_img.to(DEVICE).unsqueeze(0)
                hsi_img = hsi_img.to(DEVICE).unsqueeze(0)
                mask = mask.to(DEVICE)

                softmax = nn.Softmax(dim=1)

                preds = torch.argmax(softmax(model(rgb_img.float(), hsi_img)),axis=1).to('cpu').squeeze(0)

            # convert torch tensor to numpy array
            prediction_all_images[:,:,n] = preds.numpy()
            ground_truth_all_images[:,:,n] = mask.to('cpu').numpy()

            #calculate dice and iou score
            is_list, u_list=intersection_and_union_all_classes(mask, preds, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, preds, SINGLE_PREDICTION=True)


            if visualize == True:
            #    if data_source=='rgb':
            #        visualize_prediction_vs_ground_truth_overlay_all_sources(rgb_img.squeeze(0), None, mask, preds.squeeze(0), data_source)
            #    if data_source=='hsi':
            #        visualize_prediction_vs_ground_truth_overlay_all_sources(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), data_source)
            #    elif data_source=='sf':
            #        visualize_prediction_vs_ground_truth_overlay_all_sources(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), data_source)

                visualize_prediction_vs_ground_truth_overlay_all_sources(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), data_source)

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

            prediction_flat[0] = 8

            print(gt_flat.shape)
            print(prediction_flat.shape)
            print(CLASSES_LONG_9)

            fig, ax = plt.subplots(figsize=(10, 8))

            conf=ConfusionMatrixDisplay.from_predictions(gt_flat, prediction_flat, display_labels=CLASSES_LONG, normalize=norm_mode, 
                                                         ax=ax, xticks_rotation='vertical')

            plt.show()

            return test_ds_union, test_ds_intersection, test_ds_numerator, test_ds_denominator, img_iou, img_dice
