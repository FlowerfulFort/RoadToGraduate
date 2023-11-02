from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

def tif_to_png(input_path, output_path):
    try:
        # .tif 이미지 열기
        with Image.open(input_path) as img:
            # .png로 변환하여 저장
            img.save(output_path, 'PNG')
        print(f"이미지 변환이 완료되었습니다: {output_path}")
    except Exception as e:
        print(f"오류 발생: {e}")

# .tif 파일 경로 및 이름 설정
input_file = "./data/K3A_20190120043337_21100_00324917_L1G.tif"

# .png 파일 경로 및 이름 설정
output_file = "./data/K3A_20190120043337_21100_00324917_L1G.png"

# 변환 함수 호출
tif_to_png(input_file, output_file)