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

#color values for complete classes
num_unique_values_long = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors_long = ['black',          'white', 'green',  'red',  'cyan',          'blue',      'darkred',     'pink',     'navy', 'orange', ]
classes_long = ['background', 'chicken_front', 'chicken_back', 'Blood', 'Bones', 'SurfaceDefect', 'Discoloring', 'Scalding', 'Deformed', 'Fat/Skin']
txt_colors_long=['\033[0mblack', '\033[94mwhite', '\033[32mgreen','\033[91mred', 
                 '\033[96mcyan', '\033[94mblue', '\\033[31mdarkred',
                 '\033[95mpink' ,'\033[34mnavy' , '\033[38;2;255;165;0morange']

# model training with two possible loss functions
def model_training_multiloss(model, train_loader, val_loader, num_epochs, ce_loss_fn, dice_loss_fn, optimizer, scaler, scheduler, activate_scheduler=True):
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
        for batch_idx, (img, mask) in train_loop:
            
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
            '''
            for k in range(TRAIN_BATCH_SIZE):
            
                #calculate batch iou
                pred_combined_mask=process_prediction_to_combined_mask(predictions)
                
                #batch iou
                train_batch_iou=train_batch_iou+calculate_img_iou(iou_all_classes(pred_combined_mask[k,:,:], mask[k,:,:]))
            '''
    
        #calculate average loss
        print(f'Average Train Batch Loss: {train_batch_loss/TRAIN_BATCH_SIZE:.4f}')
        #print(f'Average Train Batch IoU: {train_batch_iou/TRAIN_BATCH_SIZE}')
        avg_train_loss_list.append(train_batch_loss/TRAIN_BATCH_SIZE)
        #avg_train_iou_list.append(train_batch_iou/TRAIN_BATCH_SIZE)
        
        ####################################################
        ############## validation instance #################
        ####################################################
        
        model.eval()
        val_loop = tqdm(enumerate(val_loader),total=len(val_loader))
        for batch_idx, (img, mask) in val_loop:
            
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
        '''    
            for k in range(VAL_BATCH_SIZE):
            
                #calculate batch iou
                pred_combined_mask=process_prediction_to_combined_mask(predictions)
                
                #batch iou
                val_batch_iou=val_batch_iou+calculate_img_iou(iou_all_classes(pred_combined_mask[k,:,:], mask[k,:,:]))
                #print(f'Validation Batch IoU: {val_batch_iou}')
            
        '''
        print(f'Average Validation Batch Loss: {val_batch_loss/VAL_BATCH_SIZE:.4f}')
        #print(f'Average Validation Batch IoU: {val_batch_iou/VAL_BATCH_SIZE:.4f}')            
        avg_val_loss_list.append(val_batch_loss/VAL_BATCH_SIZE)
        #avg_val_iou_list.append(val_batch_iou/VAL_BATCH_SIZE)
        
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
            
            ###
            # IoU
            ###
            
            '''
            axs[1].plot(range(avg_train_iou_list), avg_train_iou_list, marker='o', linestyle='-', label='Training IoU', color='blue')
            axs[1].plot(range(avg_val_iou_list), avg_val_iou_list, marker='o', linestyle='-', label='Validation IoU', color='orange')
            
            # Add labels and title
            axs[1].xlabel('Epochs')
            axs[1].ylabel('Loss')
            axs[1].title('Training vs Validation IoU')        
            
            plt.legend()
            plt.grid(True)
            plt.show()
            '''
        if epoch==50 or epoch==75 or epoch==(num_epochs-1):
            # Save all the elements to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scaler_state_dict': scaler.state_dict()
            }, f'model_e{epoch}.pt')
            
    return model, loss

# model training with one loss function
def model_training(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, scaler, scheduler, activate_scheduler=False):
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
        for batch_idx, (img, mask) in train_loop:
            
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
            '''
            for k in range(TRAIN_BATCH_SIZE):
            
                #calculate batch iou
                pred_combined_mask=process_prediction_to_combined_mask(predictions)
                
                #batch iou
                train_batch_iou=train_batch_iou+calculate_img_iou(iou_all_classes(pred_combined_mask[k,:,:], mask[k,:,:]))
            '''
    
        #calculate average loss
        print(f'Average Train Batch Loss: {train_batch_loss/TRAIN_BATCH_SIZE:.4f}')
        #print(f'Average Train Batch IoU: {train_batch_iou/TRAIN_BATCH_SIZE}')
        avg_train_loss_list.append(train_batch_loss/TRAIN_BATCH_SIZE)
        #avg_train_iou_list.append(train_batch_iou/TRAIN_BATCH_SIZE)
        
        ####################################################
        ############## validation instance #################
        ####################################################
        
        model.eval()
        val_loop = tqdm(enumerate(val_loader),total=len(val_loader))
        for batch_idx, (img, mask) in val_loop:
            
            with torch.no_grad():
                img = img.to(DEVICE)
                mask = mask.to(DEVICE)
                #mask = mask
            
                # forward
                with torch.cuda.amp.autocast():
                    predictions = model(img.float())
                    val_loss = loss_fn(predictions, mask.type(torch.long))
    
            # update tqdm loop
            val_loop.set_postfix(val_loss=val_loss.item())
            
            val_batch_loss = val_batch_loss + val_loss.item()
        '''    
            for k in range(VAL_BATCH_SIZE):
            
                #calculate batch iou
                pred_combined_mask=process_prediction_to_combined_mask(predictions)
                
                #batch iou
                val_batch_iou=val_batch_iou+calculate_img_iou(iou_all_classes(pred_combined_mask[k,:,:], mask[k,:,:]))
                #print(f'Validation Batch IoU: {val_batch_iou}')
            
        '''
        print(f'Average Validation Batch Loss: {val_batch_loss/VAL_BATCH_SIZE:.4f}')
        #print(f'Average Validation Batch IoU: {val_batch_iou/VAL_BATCH_SIZE:.4f}')            
        avg_val_loss_list.append(val_batch_loss/VAL_BATCH_SIZE)
        #avg_val_iou_list.append(val_batch_iou/VAL_BATCH_SIZE)
        
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
            
            ###
            # IoU
            ###
            
            '''
            axs[1].plot(range(avg_train_iou_list), avg_train_iou_list, marker='o', linestyle='-', label='Training IoU', color='blue')
            axs[1].plot(range(avg_val_iou_list), avg_val_iou_list, marker='o', linestyle='-', label='Validation IoU', color='orange')
            
            # Add labels and title
            axs[1].xlabel('Epochs')
            axs[1].ylabel('Loss')
            axs[1].title('Training vs Validation IoU')        
            
            plt.legend()
            plt.grid(True)
            plt.show()
            '''
        if epoch==50 or epoch==75 or epoch==(num_epochs-1):
            # Save all the elements to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'scaler_state_dict': scaler.state_dict()
            }, f'model_e{epoch}_{CURRENT_DATE}.pt')
            
    return model, loss


def rgb_visualize_prediction_vs_ground_truth_single_batches_before_argmax(model, loader, height, width):
    
    model.eval()
    
    #for checking if masks fit to respective image, all defects are displayed in a unique color
    
    print('Legend:')
    for i, color in enumerate(colors_long):
        print(f'{txt_colors_long[i]} -> {classes_long[i]}')
        
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
        for c in range(len(num_unique_values_long)):
            #normalize probability of predicted masks
            #prob_pred_mask[j,c,:,:]=prob_pred_mask[j,c,:,:]/torch.max(prob_pred_mask[j,c,:,:])
            prob_masks.append(prob_pred_mask[j,c,:,:])
        
        # Initialize an empty overlay
        overlay = np.zeros((height, width, 3))  
        for i, prob_mask in enumerate(prob_masks):

            color = np.array(plt.cm.colors.to_rgba(colors_long[i])[:3])  # Get color for class
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
    
        #for k in range(3):  # Undo normalization for each channel
        #    denorm_img[:, :, k] = single_img[:, :, k] * img_std[k] + img_mean[k]

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
    
    model.eval()

    if is_img_normalized:

        fig , axs =  plt.subplots(1, 3, figsize=(18, 12))

        denorm_img = img.numpy().transpose((1, 2, 0)).copy()

        for j in range(3):  # Undo normalization for each channel
            denorm_img[:, :, j] = denorm_img[:, :, j] * img_std[j] + img_mean[j]

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
