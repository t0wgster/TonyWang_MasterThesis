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
