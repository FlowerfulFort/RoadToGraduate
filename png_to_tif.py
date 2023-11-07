from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def extract_tags_from_tif(tif_file):
    try:
        with Image.open(tif_file) as img:
            tags = img.tag_v2  # 태그 정보를 가져옵니다.
            return tags
    except Exception as e:
        print(f"태그를 추출하는 동안 오류 발생: {e}")
        return None

def create_tif_with_tags(tif_file, png_file, output_file):
    try:
        with Image.open(tif_file) as tif_img:
            with Image.open(png_file) as png_img:
                # tif 이미지의 태그를 가져옵니다.
                tags = extract_tags_from_tif(tif_file)

                # 새로운 tif 파일을 만들기 위해 png 이미지를 tif 이미지의 크기와 모드로 변환합니다.
                png_img = png_img.convert(tif_img.mode)
                png_img = png_img.resize(tif_img.size)

                # 새로운 tif 이미지에 png 이미지를 붙입니다.
                tif_img.paste(png_img, (0, 0))

                # 새로운 tif 이미지에 이전의 태그를 할당합니다.
                tif_img.tag_v2 = tags

                # tif 파일로 저장합니다.
                tif_img.save(output_file, compression="tiff_deflate")
                print(f"새로운 TIF 파일 생성: {output_file}")
    except Exception as e:
        print(f"TIF 파일 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    tif_file_path = './data/K3A_20190120043337_21100_00324917_L1G.tif'  # 기존의 tif 파일 경로
    png_file_path = './data/K3A_20190120043337_21100_00324917_L1G_mask.png'  # 추가할 png 파일 경로
    output_tif_path = './data/K3A_20190120043337_21100_00324917_L1G_mask.tif'  # 새로운 tif 파일 경로

    tags = extract_tags_from_tif(tif_file_path)
    if tags:
        print("추출된 태그:")
        for tag, value in tags.items():
            print(f"{tag}: {value}")

    create_tif_with_tags(tif_file_path, png_file_path, output_tif_path)
