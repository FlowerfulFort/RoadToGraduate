import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
import glob
import numpy as np
from augmentations.transforms import get_flips_colors_augmentation
import shutil
from PIL import Image
from dataset.reading_image_provider import ReadingImageProvider
from dataset.raw_image import RawImageType
from pytorch_utils.train import train
from pytorch_utils.concrete_eval import CropEvaluator
from utils import update_config, get_csv_folds
import argparse
import json
from config import Config

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('--fold', type=int)
parser.add_argument('--training', action='store_true')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
config = Config(**cfg)
skip_folds = []

if args.fold is not None:
    skip_folds = [i for i in range(4) if i != int(args.fold)]
    
test = not args.training

config = update_config(config, dataset_path=os.path.join('/opt', 'datasets', 'full'))
    
paths = {
    'masks': os.path.join(config.dataset_path, 'masks2m'),
    #'images': os.path.join(config.dataset_path, 'large_tif')
    'images': os.path.join(config.dataset_path, 'images')
}
fn_mapping = {
    'masks': lambda name: os.path.splitext(name)[0] + '.png'
}

image_suffix=None

class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        #return self.reflect_border(data, 22)
        return data

def change_image_channels(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = image[:, :, :-1]
    cv2.imwrite(image_path, image)
    
def eval_roads():
    global config
    # rows, cols = 1344, 1344
    rows, cols = 512, 512
    config = update_config(config, target_rows=rows, target_cols=cols)
    ds = ReadingImageProvider(RawImageType, paths, fn_mapping, image_suffix=image_suffix)
    # ds = ReadingImageProvider(RawImageTypePad, paths, fn_mapping, image_suffix=image_suffix)

    folds = [([], list(range(len(ds)))) for i in range(4)]
    # num_workers = 0 if os.name == 'nt' else 2
    # num_workers=0
    num_workers=16
    keval = CropEvaluator(config, ds, test=test, flips=3, num_workers=num_workers, border=0)
    for fold, (t, e) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        keval.predict(fold, e)
        break
        
def pre():
    tif_directory = "/opt/datasets/full/large_tif"
    png_directory = "/opt/datasets/full/images"
    
    tif_files = glob.glob(os.path.join(tif_directory, "*.tif"))
    
    chunk_width = 5120
    chunk_height = 5120
    
    tif_files = [f for f in os.listdir(tif_directory) if f.endswith(".tif")]
    
    for tif_file in tif_files:
        tif_image = cv2.imread(os.path.join(tif_directory, tif_file), cv2.IMREAD_UNCHANGED)
        image_height, image_width = tif_image.shape[0], tif_image.shape[1]
        final_image = tif_image.copy()
        
        for x in range(0, image_width, chunk_width):
            for y in range(0, image_height, chunk_height):
                chunk = tif_image[y:y+chunk_height, x:x+chunk_width]
                chunk_filename = f"{tif_file[:-4]}_{x}_{y}.png"
                cv2.imwrite(os.path.join(png_directory, chunk_filename), chunk)

    for filename in os.listdir(png_directory):
        if filename.endswith(".png"):
            image_path = os.path.join(png_directory, filename)
            change_image_channels(image_path)
    
def post():
    image_folder = os.path.abspath('.')
    subdirectories = [os.path.join(image_folder, d) for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]

    for subdirectory in subdirectories:
        base_name = os.path.basename(subdirectory) + "_mask"
        images_by_y = {}
        image_files = [f for f in os.listdir(subdirectory) if f.endswith('.png')]

        for image_file in image_files:
            image_path = os.path.join(subdirectory, image_file)
            image = cv2.imread(image_path)

            x = int(image_file.split('_')[-2: -1][0].split('.')[0])
            y = int(image_file.split('_')[-1: ][0].split('.')[0])

            if y in images_by_y:
                images_by_y[y].append((x, image))
            else:
                images_by_y[y] = [(x, image)]

        vertical_combined_images = []
            
        for y in sorted(images_by_y.keys()):
            images_sorted_by_x = sorted(images_by_y[y], key=lambda x: x[0])
            vertical_combined_image = np.hstack([cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for _, img in images_sorted_by_x])
            vertical_combined_images.append(vertical_combined_image)

        if vertical_combined_images:
            combined_image = np.vstack(vertical_combined_images)
            output_path = os.path.join(image_folder, f"{base_name}.png")
            cv2.imwrite(output_path, combined_image)

    directory_path = "/opt/datasets/full/images"

    try:
        file_list = os.listdir(directory_path)
        for filename in file_list:
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                pass
    except Exception as e:
        print(f"오류 발생: {e}")
"""           
    try:
        shutil.rmtree(image_folder)
    except Exception as e:
        print(f"오류 발생: {e}")
"""
if __name__ == "__main__":
    print('eval start')
    pre()
    eval_roads()
    post()
    