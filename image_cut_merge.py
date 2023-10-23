import cv2
print(cv2.__version__)
import os

# 입력 이미지 경로
input_image_path = "input_image.png"

# 저장할 패치 이미지들을 담을 디렉토리 경로
output_patch_dir = "output_patches"
os.makedirs(output_patch_dir, exist_ok=True)

# 패치 크기 정의
patch_size = (512, 512)

def crop_image_and_save(input_image_path, output_patch_dir):
    # 이미지를 불러옵니다.
    image = cv2.imread(input_image_path)

    # 이미지의 높이와 너비 가져오기
    height, width, _ = image.shape

    # 패치 크기로 이미지를 잘라냅니다.
    patch_count = 0
    for y in range(0, height, patch_size[1]):
        for x in range(0, width, patch_size[0]):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]

            # 패치를 별도의 파일로 저장 (png 형식)
            patch_filename = f"patch_{patch_count}.png"
            patch_path = os.path.join(output_patch_dir, patch_filename)
            cv2.imwrite(patch_path, patch, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            patch_count += 1

# 함수를 호출하여 이미지를 자르고 패치를 저장합니다.
crop_image_and_save(input_image_path, output_patch_dir)