# import torch
# from torchvision import transforms
# from PIL import Image
# import sys
# import os
# import torch
# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# from augmentations.transforms import get_flips_colors_augmentation

# from dataset.reading_image_provider import ReadingImageProvider
# from dataset.raw_image import RawImageType
# from pytorch_utils.train import train
# from pytorch_utils.concrete_eval import FullImageEvaluator
# from utils import update_config, get_csv_folds
# import argparse
# import json
# from config import Config

# model = resnet34()

# transform = transforms.Compose([
#     transforms.Resize((512, 512)
#     transforms.ToTensor(),          
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
# ])

# img_path = os.path.join(datasets, 'images')
# image = Image.open(image_path)

# input_tensor = transform(image)
# input_tensor = input_tensor.unsqueeze(0)

# model.eval()

# with torch.no_grad():
#     output = model(input_tensor)

# print("모델 아웃풋 크기:", output.shape

import torch
from torchsummary import summary
from pytorch_zoo import unet

model = {
    'resnet34': unet.Resnet34_upsample,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
summary(model.to(device), (3, 512, 512))