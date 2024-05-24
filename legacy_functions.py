
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

