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

from TonyWang_MasterThesis.functions_and_constants import *
from TonyWang_MasterThesis.visualisation_and_evaluation import *



def capture_model_metrics_pixelwise_and_confusion_matrix_sf_postprocess_cv(model, test_dataset_final, data_source, kernel_size, visualize = True, 
                                                         confusion_matrix = True, norm_mode = 'pred', mask_shape=(320,320),
                                                         smooth=1e-8, ):
    '''
    Description: Specifically for Post processing. Predicts Mask and Calculate the IoU and Dice Score of each image within a dataset for all data sources. Individually output 
    those scores for each individual class. 
    Input: Model, Test Dataset, Data Source (String)
    Ouput: Intersection List, Union List, Dice Num List, Dice Denom List
    '''
    test_ds_union = [0,0,0,0,0,0,0,0,0,0]
    test_ds_intersection = [0,0,0,0,0,0,0,0,0,0]
    test_ds_numerator = [0,0,0,0,0,0,0,0,0,0]
    test_ds_denominator = [0,0,0,0,0,0,0,0,0,0]
    img_dice = []
    img_iou = []
    kernel = np.ones((kernel_size,kernel_size),np.uint8)                                                       

    #create array to capture all ground truth and predictions to calculate final IoU and Dice Score at the end                                                         
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

            # get post processed mask
            preds_np = preds.squeeze(0).numpy().astype(np.uint8)
            postprocessed_pred_mask = cv2.morphologyEx(preds_np, cv2.MORPH_OPEN, kernel)
            mask_np = mask.cpu().numpy().astype(np.uint8)
            postprocessed_truth_mask = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)

            # add current mask and prediction to stacked array for 
            prediction_all_images[:,:,n] = postprocessed_pred_mask
            ground_truth_all_images[:,:,n] = postprocessed_truth_mask

            #calculate dice and iou score for calculating final IoU and Dice Score at the end    
            is_list, u_list=intersection_and_union_all_classes(mask, preds, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, preds, SINGLE_PREDICTION=True)

            if visualize == True:

                visualize_prediction_vs_ground_truth_overlay_all_sources_postprocessing(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), postprocessed_pred_mask, data_source)

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



def visualize_prediction_vs_ground_truth_overlay_all_sources_postprocessing(rgb_img, hsi_img, truth_mask, pred_mask, processed_pred_mask, data_source):
    '''
    Description: Specifically for Post processing. Visualizes model results and during visualisation overlays the mask on top of the image. For all data sources
    Input: RGB Image, HSI Image, Ground Truth Mask, Predicted Mask, Data Source
    '''
    if data_source == 'rgb':

        fig , axs =  plt.subplots(1, 4, figsize=(18, 12))

        axs[0].set_title('Plain Image RGB')
        axs[1].set_title('Image with Ground Truth')
        axs[2].set_title('Image with Prediction')
        axs[3].set_title('Image with Post Processing')

        axs[0].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[1].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[2].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[3].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))

        axs[1].imshow(np.asarray(truth_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[2].imshow(np.asarray(pred_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[3].imshow(processed_pred_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)

        for ax in axs: 
            ax.axis('off')


    elif data_source == 'hsi':

        fig , axs =  plt.subplots(1, 5, figsize=(18, 12))

        axs[0].set_title('Plain Image HSI')
        axs[1].set_title('Image with Ground Truth')
        axs[2].set_title('Image with Prediction')
        axs[3].set_title('Image with Post Processing')
        axs[4].set_title('RGB Reference')

        axs[0].imshow(np.asarray(hsi_img.to('cpu').permute(1,2,0))[:,:,0])
        axs[1].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[2].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[3].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[4].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))

        axs[1].imshow(np.asarray(truth_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[2].imshow(np.asarray(pred_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[3].imshow(processed_pred_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)

        for ax in axs: 
            ax.axis('off')
  
    elif data_source == 'sf':

        fig , axs =  plt.subplots(1, 5, figsize=(18, 12))

        axs[0].set_title('Plain Image RGB')
        axs[1].set_title('Plain Image HSI')
        axs[2].set_title('Image with Ground Truth')
        axs[3].set_title('Image with Prediction')
        axs[4].set_title('Image with Post Processing')

        axs[0].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[1].imshow(np.asarray(hsi_img.to('cpu').permute(1,2,0))[:,:,0])
        axs[2].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[3].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))
        axs[4].imshow(np.asarray(rgb_img.to('cpu').permute(1,2,0)))

        axs[2].imshow(np.asarray(truth_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[3].imshow(np.asarray(pred_mask.to('cpu')), cmap=cmap_long, norm=norm_long, alpha=0.3)
        axs[4].imshow(processed_pred_mask, cmap=cmap_long, norm=norm_long, alpha=0.3)

        for ax in axs: 
            ax.axis('off')

    plt.show()

#### Probability based post processing
def probability_based_kernel_post_processing(model, smooth, dataset, mask_shape, kernel_size):

    #create array to capture all ground truth and predictions to calculate final IoU and Dice Score at the end
    ground_truth_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))
    prediction_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))

    with torch.no_grad():
        model.eval()
        for n, batch in enumerate(dataset):

            #empty numpy mask to fit the one hot encoded classes
            pp_one_hot_pred_masks = np.zeros((384,320, 10))

            rgb_img, hsi_img, mask = batch

            #predict imgs in dataset
            rgb_img = rgb_img.to(DEVICE).unsqueeze(0)
            mask = mask.to(DEVICE)
            softmax = nn.Softmax(dim=1)

            #predicted probabilities
            preds = softmax(model(rgb_img.float())).to('cpu').squeeze(0).permute(1,2,0)
            preds_np = preds.numpy()

            #combined masks for comparison
            preds_argmax = torch.argmax(preds, axis=-1).to('cpu').squeeze(0)

            start = time.time()

            # individual kernels for each defect class
            for i in range(9, 2, -1):
                kernel = np.ones((kernel_size[i],kernel_size[i]),np.uint8)
                radius = int(kernel_size[i]/2)

                #elliptical kernel size
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

                #closing operation
                pp_one_hot_pred_masks[:,:,i] = cv2.morphologyEx(preds_np[:,:,i], cv2.MORPH_CLOSE, kernel)

            # individual kernels for each background, fillet front and back class
            for i in range(2, -1, -1):
                kernel = np.ones((kernel_size[i],kernel_size[i]),np.uint8)
                radius = int(kernel_size[i]/2)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

                #dilation operation
                pp_one_hot_pred_masks[:,:,i] = cv2.dilate(preds_np[:,:,i], kernel)

            end = time.time()
            print(f'Time: {end - start}')

            # convert back to torch because evaluation functions only work with torch tensors
            pp_one_hot_pred_masks = torch.from_numpy(pp_one_hot_pred_masks).to('cpu')

            # combine post processed masks
            single_mask = np.argmax(pp_one_hot_pred_masks, axis=-1)

            # add current mask and prediction to stacked array for
            prediction_all_images[:,:,n] = single_mask
            ground_truth_all_images[:,:,n] = mask.to('cpu').numpy()

            #calculate dice and iou score for calculating final IoU and Dice Score at the end
            is_list, u_list=intersection_and_union_all_classes(mask, single_mask, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, single_mask, SINGLE_PREDICTION=True)

            #visualize predictions vs ground truth
            visualize_prediction_vs_ground_truth_overlay_all_sources_postprocessing(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds_argmax.squeeze(0), single_mask, 'rgb')

            #print iou and dice score for each individual image
            print('IOU')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {is_list[i]/(u_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_union[i] += u_list[i]
                    test_ds_intersection[i] += is_list[i]

            print(TXT_COLORS_LONG_COLOR_ONLY[0]+ 'x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x')
            print('Dice')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {n_list[i]/(d_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_numerator[i] += n_list[i]
                    test_ds_denominator[i] += d_list[i]

                print(TXT_COLORS_LONG_COLOR_ONLY[0])

    #evaluate overall model
    calculate_model_metrics(test_ds_intersection, test_ds_union, test_ds_numerator, test_ds_denominator, defects_only=True)

##### Region based Post processing
def region_based_kernel_post_processing(model, smooth, dataset, mask_shape, kernel_size):

    #create array to capture all ground truth and predictions to calculate final IoU and Dice Score at the end
    ground_truth_all_images=np.zeros((mask_shape[0], mask_shape[1], len(dataset)))
    prediction_all_images=np.zeros((mask_shape[0], mask_shape[1], len(dataset)))
    pp_one_hot_pred_masks = np.zeros((384,320, 10),np.uint8)

    with torch.no_grad():
        model.eval()
        for n, batch in enumerate(dataset):
            rgb_img, hsi_img, mask = batch

            rgb_img = rgb_img.to(DEVICE).unsqueeze(0)
            mask = mask.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(rgb_img.float())),axis=1).to('cpu').squeeze(0)

            #one hot encoding of mask after argmax
            one_hot_pred_masks=F.one_hot(preds.to(torch.int64), num_classes=10).to(DEVICE)

            # individual kernels for each defect class
            for i in range(9, 2, -1):
                kernel = np.ones((kernel_size[i],kernel_size[i]),np.uint8)
                pp_one_hot_pred_masks[:,:,i] = cv2.morphologyEx(one_hot_pred_masks[:,:,i].to('cpu').numpy().astype(np.uint8), cv2.MORPH_OPEN, kernel)

            # individual kernels for each background, fillet front and back class
            for i in range(2, -1, -1):
                kernel = np.ones((kernel_size[i],kernel_size[i]),np.uint8)
                pp_one_hot_pred_masks[:,:,i] = cv2.morphologyEx(one_hot_pred_masks[:,:,i].to('cpu').numpy().astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            #give defect class and fillet front and back more weight than background class
            pp_one_hot_pred_masks[:,:,1:3] = pp_one_hot_pred_masks[:,:,1:3]*2
            pp_one_hot_pred_masks[:,:,3:10] = pp_one_hot_pred_masks[:,:,3:10]*3

            #combine mask
            single_mask_array = np.argmax(pp_one_hot_pred_masks, axis=-1)
            single_mask = torch.from_numpy(single_mask_array)

            # add current mask and prediction to stacked array for confusion matrix
            prediction_all_images[:,:,n] = single_mask_array
            ground_truth_all_images[:,:,n] = mask.to('cpu').numpy()

            #calculate dice and iou score for calculating final IoU and Dice Score at the end
            is_list, u_list=intersection_and_union_all_classes(mask, single_mask, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, single_mask, SINGLE_PREDICTION=True)

            #visualize predictions vs ground truth
            visualize_prediction_vs_ground_truth_overlay_all_sources_postprocessing(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), single_mask, 'rgb')

            #print iou and dice score for each individual image
            print('IOU')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {is_list[i]/(u_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_union[i] += u_list[i]
                    test_ds_intersection[i] += is_list[i]

            print(TXT_COLORS_LONG_COLOR_ONLY[0]+ 'x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x')
            print('Dice')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {n_list[i]/(d_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_numerator[i] += n_list[i]
                    test_ds_denominator[i] += d_list[i]

            print(TXT_COLORS_LONG_COLOR_ONLY[0])

    calculate_model_metrics(test_ds_intersection, test_ds_union, test_ds_numerator, test_ds_denominator, defects_only=True)

##### CRF based Post processing
def crf_based_post_processing(model, smooth, dataset, mask_shape, theta_a, theta_b, theta_g):

    #create array to capture all ground truth and predictions to calculate final IoU and Dice Score at the end
    ground_truth_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))
    prediction_all_images=np.zeros((mask_shape[0], mask_shape[1], len(test_dataset_final)))
    pp_one_hot_pred_masks = np.zeros((384,320, 10),np.uint8)

    #model inference
    with torch.no_grad():
        model.eval()
        for n, batch in enumerate(test_dataset_final):
            rgb_img, hsi_img, mask = batch

            rgb_img = rgb_img.to(DEVICE).unsqueeze(0)
            mask = mask.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(rgb_img.float())),axis=1).to('cpu').squeeze(0)

            #convert original img and annotated img into numpy arrays
            original_image=rgb_img.to('cpu').squeeze().permute(1,2,0).numpy()
            annotated_image=preds.to('cpu').numpy().astype(np.uint32)

            #from here: code snippets from git@github.com:lucasb-eyer/pydensecrf.git
            #and from here: code snippets from git@github.com:dhawan98/Post-Processing-of-Image-Segmentation-using-CRF.git
            #number of classes in dataset
            n_labels_a = 10

            #flatten segmentation mask
            labels_a = annotated_image.flatten()

            #Setting up the CRF model
            d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels_a)

            # get unary potentials (neg log probability)
            U = unary_from_labels(labels_a, n_labels_a, gt_prob=0.90, zero_unsure=False)

            #calculate Gibbs energy
            d.setUnaryEnergy(U)

            # This adds the color-independent term, features are the locations only.
            # smoothing kernel
            d.addPairwiseGaussian(sxy=(theta_gamma, theta_gamma), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
            # appearance kernel
            d.addPairwiseBilateral(sxy=(theta_alpha, theta_alpha), srgb=(theta_beta, theta_beta, theta_beta), rgbim=original_image.astype(np.uint8),
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

            #Run CRF model inference for x steps
            Q = d.inference(1)

            # Find out the most probable class for each pixel.
            MAP = np.argmax(Q, axis=0)

            # Convert the MAP (labels) back to the corresponding colors and save the image.
            post_processed_mask=MAP.reshape(annotated_image.shape)

            ####
            #code snippets from github repos end here

            #convert mask back to torch tensor for evaluating purposes
            single_mask = torch.from_numpy(post_processed_mask)

            #calculate dice and iou score for calculating final IoU and Dice Score at the end
            is_list, u_list=intersection_and_union_all_classes(mask, single_mask, SINGLE_PREDICTION=True)
            n_list, d_list=dice_values_all_classes(mask, single_mask, SINGLE_PREDICTION=True)

            #visualize predictions vs ground truth
            visualize_prediction_vs_ground_truth_overlay_all_sources_postprocessing(rgb_img.squeeze(0), hsi_img.squeeze(0), mask, preds.squeeze(0), single_mask, 'rgb')

            #print iou and dice score for each individual image
            print('IOU')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {is_list[i]/(u_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_union[i] += u_list[i]
                    test_ds_intersection[i] += is_list[i]

            print(TXT_COLORS_LONG_COLOR_ONLY[0]+ 'x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x')
            print('Dice')
            for i in range(len(NUM_UNIQUE_VALUES_LONG)):
                if is_ground_truth_empty(mask)[i] and is_prediction_empty(single_mask)[i]:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: Empty')
                else:
                    print(f'{TXT_COLORS_LONG_COLOR_ONLY[i]} - {CLASSES_LONG[i]}: {n_list[i]/(d_list[i]+smooth):.4f}')
                    #tracking class average of iou across all images
                    test_ds_numerator[i] += n_list[i]
                    test_ds_denominator[i] += d_list[i]

            print(TXT_COLORS_LONG_COLOR_ONLY[0])

        calculate_model_metrics(test_ds_intersection, test_ds_union, test_ds_numerator, test_ds_denominator, defects_only=True)
