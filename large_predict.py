import os
import cv2
import glob
import shutil
import numpy as np
from PIL import Image
from config import Config
from utils import update_config
from dataset.raw_image import RawImageType
from pytorch_utils.concrete_eval import CropEvaluator
from dataset.reading_image_provider import ReadingImageProvider

#만든 함수들
from image_to_wkt import convert_image_to_wkt

# 이미지 채널 변경
def change_image_channels(target_file):
    image = cv2.imread(target_file, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = image[:, :, :-1]
    cv2.imwrite(target_file, image)

# 이미지 전처리
def preprocess_image(image_path, intermediate_dir, image_name):
    tif_directory = os.path.dirname(os.path.abspath(image_path))
    png_directory = os.path.join(intermediate_dir, image_name)
    tif_files = glob.glob(os.path.join(tif_directory, "*.tif"))
    
    chunk_width = 5120
    chunk_height = 5120
    
    if not os.path.exists(png_directory):
        os.makedirs(png_directory)
    
    for tif_file in tif_files:
        tif_image = cv2.imread(os.path.join(tif_directory, tif_file), cv2.IMREAD_UNCHANGED)
        image_height, image_width = tif_image.shape[0], tif_image.shape[1]
        final_image = tif_image.copy()
        
        for x in range(0, image_width, chunk_width):
            for y in range(0, image_height, chunk_height):
                chunk = tif_image[y:y+chunk_height, x:x+chunk_width]
                chunk_filename = f"{image_name}_{x}_{y}.png"
                cv2.imwrite(os.path.join(png_directory, chunk_filename), chunk)

    for filename in os.listdir(png_directory):
        if filename.endswith(".png"):
            target_file = os.path.join(png_directory, filename)
            change_image_channels(target_file)

# 이미지 후처리
def postprocess_image(intermediate_dir, image_name):
    target_dir = os.path.join(intermediate_dir, f"{image_name}_mask")
    output_path = os.path.join(intermediate_dir, f"{image_name}_mask.png")
    
    images_by_y = {}
    image_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]

    for image_file in image_files:
        image_path_ = os.path.join(target_dir, image_file)
        image = cv2.imread(image_path_)

        x = int(image_file.split('_')[-2: -1][0].split('.')[0])
        y = int(image_file.split('_')[-1: ][0].split('.')[0])
        images_by_y.setdefault(y, []).append((x, image))

    vertical_combined_images = [np.hstack([cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for _, img in sorted(images, key=lambda x: x[0])]) for y, images in sorted(images_by_y.items())]
            
    if vertical_combined_images:
        combined_image = np.vstack(vertical_combined_images)
        cv2.imwrite(output_path, combined_image)

def remove_directory(directory_path):
    try:
        # 디렉토리가 존재하는지 확인
        if os.path.exists(directory_path):
            # 디렉토리 안이 비어있는지 확인
            if not os.listdir(directory_path):
                os.rmdir(directory_path)
            else:
                # 디렉토리가 비어있지 않다면 강제로 제거
                shutil.rmtree(directory_path)
        else:
            print(f"디렉토리 '{directory_path}'가 존재하지 않습니다.")
    except Exception as e:
        print(f"디렉토리 제거 중 오류가 발생했습니다: {e}")

# 메인 함수
def main(image_path, intermediate_dir, image_name, config, args):
    print('eval start')
    dataset_dir = os.path.join(os.path.dirname(image_path))
    image_dir = os.path.join(*image_path.split('/')[:-1])
    final_output = os.path.join(image_dir, f"{image_name}.txt") #최종 산물 저장 경로, 현재는 원본 이미지와 같은 경로
            
    preprocess_image(image_path, intermediate_dir, image_name)

    rows, cols = 512, 512
    config = update_config(config, target_rows=rows, target_cols=cols)
    image_suffix=None
    fn_mapping = {'masks': lambda name: os.path.splitext(name)[0] + '.png'}
    ds = ReadingImageProvider(RawImageType,os.path.join(intermediate_dir, image_name), fn_mapping, image_suffix=image_suffix)
    
    folds = [([], list(range(len(ds)))) for i in range(4)]
    num_workers = int(os.getenv('EVAL_WORKER', '16'))
    keval = CropEvaluator(config, ds, test=not args.training, flips=3, num_workers=num_workers, border=0)
    
    for fold, (_, e) in enumerate(folds):
        if args.fold is not None and int(args.fold) != fold:
            continue
        keval.predict(intermediate_dir, image_name, fold, e)
        break
    postprocess_image(intermediate_dir, image_name)
    convert_image_to_wkt(image_path, intermediate_dir, final_output, image_name)

                                  
if __name__ == "__main__":
    import argparse
    import json

    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--training', action='store_true')
    args = parser.parse_args()

    my_path = os.path.dirname(os.path.realpath(__file__))
    with open(f'{my_path}/resnet34_512_02_02.json', 'r') as f:
        cfg = json.load(f)
    # with open('./resnet34_512_02_02.json', 'r') as f:
    #     cfg = json.load(f)
    
    config = Config(**cfg)
    config = update_config(config, dataset_path=os.path.dirname(os.path.abspath(args.image_path)))
    
    # 중간 산물 저장 경로 원본 이미지와 같은 경로에 만들어지고 없어진다
    intermediate_dir = os.path.join(os.path.dirname(args.image_path), "intermediate")
    image_name = args.image_path.split('/')[-1][:-4]
    
    if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)

    main(args.image_path, intermediate_dir, image_name, config, args)
    
    remove_directory(intermediate_dir)