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
