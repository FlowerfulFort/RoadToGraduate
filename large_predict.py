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
parser.add_argument('image_path')
parser.add_argument('--fold', type=int)
parser.add_argument('--training', action='store_true')
args = parser.parse_args()
with open('./resnet34_512_02_02.json', 'r') as f:
    cfg = json.load(f)
config = Config(**cfg)
skip_folds = []

config = update_config(config, dataset_path=os.path.dirname(os.path.abspath(args.image_path)))
dataset_dir = args.image_path.split('/')[-1].split('.')[0] + "_pre"

paths = {
    'images': os.path.join(config.dataset_path, dataset_dir)
}
fn_mapping = {
    'masks': lambda name: os.path.splitext(name)[0] + '.png'
}

if args.fold is not None:
    skip_folds = [i for i in range(4) if i != int(args.fold)]

test = not args.training
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
    
def eval_roads(image_path):
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
        
def pre(image_path):
    tif_directory = os.path.dirname(os.path.abspath(image_path))
    image_name = os.path.join(tif_directory , image_path).split('/')[-1].split('.')[0] + "_pre"
    png_directory = os.path.join(tif_directory, image_name)
    
    if not os.path.exists(png_directory):
            os.makedirs(png_directory)

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

def post(image_path):
    image_folder = os.path.dirname(os.path.abspath(image_path))
    dir_ = args.image_path.split('/')[-1].split('.')[0]
    
    base_name = os.path.basename(dir_) + "_mask"
    images_by_y = {}
    image_files = [f for f in os.listdir(dir_) if f.endswith('.png')]

    for image_file in image_files:
        image_path_ = os.path.join(dir_, image_file)
        image = cv2.imread(image_path_)

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
        
    pre_path = os.path.join(os.path.dirname(os.path.abspath(image_path)), os.path.basename(dir_) + "_pre")
    mask_path = os.path.join(os.path.dirname(image_folder), dir_)
    try:
        file_list = os.listdir(pre_path)
        for filename in file_list:
            file_path = os.path.join(pre_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                pass
        os.rmdir(pre_path)

        file_list = os.listdir(mask_path)
        for filename in file_list:
            file_path = os.path.join(mask_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                pass
        os.rmdir(mask_path)
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    if args.image_path is not None:
        print('eval start')
        pre(args.image_path)
        eval_roads(args.image_path)
        post(args.image_path)
    