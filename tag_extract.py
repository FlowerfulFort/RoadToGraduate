from osgeo import gdal
from pyproj import Proj, Transformer
from shapely import wkt
from shapely.geometry import Point, LineString
from shapely.wkt import loads, dumps
from shapely.affinity import translate
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def extract_tags_from_tif(tif_file, output_txt):
    try:
        with Image.open(tif_file) as img:
            tags = img.tag_v2  # 태그 정보를 가져옵니다.
            with open(output_txt, 'w') as txt_file:
                for tag, value in tags.items():
                    txt_file.write(f"{tag}: {value}\n")
            print(f"태그를 {output_txt}에 저장했습니다.")
            return tags
    except Exception as e:
        print(f"태그를 추출하는 동안 오류 발생: {e}")
        return None

def extract_projection_info(geotiff_path):
    dataset = gdal.Open(geotiff_path)

    if dataset is None:
        print(f"Failed to open GeoTIFF file: {geotiff_path}")
        return None

    # 좌표 체계 정보 가져오기
    projection_info = dataset.GetProjection()

    # 데이터셋 닫기
    dataset = None

    return projection_info

def convert_coordinates(src_proj, dst_proj, coordinates):
    # 좌표 변환
    transformer = Transformer.from_proj(src_proj, dst_proj)
    lat, lon = transformer.transform(coordinates[0], coordinates[1])

    return lon, lat

def read_and_transform_linestring(input_file, output_file, src_proj, dst_proj):
    with open(input_file, 'r') as input_file:
        # 원본 좌표 체계와 대상 좌표 체계 정의
        src_proj = Proj(src_proj)
        dst_proj = Proj(dst_proj)

        # 좌표 변환 후 파일에 쓰기
        with open(output_file, 'w') as output_file:
            for line in input_file:
                # WKT에서 LineString 추출
                linestring = wkt.loads(line.strip())

                # 좌표 변환
                transformed_linestring = LineString(
                    [Point(convert_coordinates(src_proj, dst_proj, coords)) for coords in linestring.coords]
                )

                # 변환된 LineString을 새로운 파일에 쓰기
                output_file.write(f"{transformed_linestring}\n")


def extract_values_from_file(file_path, target_tag, index1, index2):
    with open(file_path, 'r') as file:
        for line in file:
            tag, value = map(str.strip, line.split(':', 1))
            
            if tag == str(target_tag):
                # 값을 괄호와 공백을 제거하고 파싱한 후, 3번째와 4번째 인덱스의 값만 추출
                values = tuple(map(float, value.strip('()').split(',')))
                return values[index1], values[index2]

def transform_and_save_wkt(input_file, output_file, transform_values, convert_value):
    # 좌표 변환값
    x_offset, y_offset = transform_values
    x_convert, y_convert = convert_value

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # WKT 파싱
            geometry = loads(line.strip())
            
            # 좌표 변환
            transformed_coordinates = [((x * x_convert) + x_offset, y_offset - (y * y_convert)) for x, y in geometry.coords]
            
            # 변환된 좌표를 새로운 WKT로 저장
            new_wkt = f"{geometry.geom_type.upper()} ({', '.join([f'{x} {y}' for x, y in transformed_coordinates])})"
            outfile.write(new_wkt + '\n')
"""     
# GeoTIFF 파일 경로를 지정
geotiff_path = "./data/K3A_20190120043337_21100_00324917_L1G.tif"
output_path = "./data/K3A_20190120043337_21100_00324917_L1G_tags.txt"

# geotiff_path = "./K3A_20151028043641_03268_00099058_L1G.tif"
# output_path = "./K3A_20151028043641_03268_00099058_L1G_tags.txt"

# geotiff_path = "./K3A_20190102043431_20828_00349927_L1G.tif"
# output_path = "./K3A_20190102043431_20828_00349927_L1G_tags.txt"
target_tag1 = 33922
target_tag2 = 33550

extract_tags_from_tif(geotiff_path, output_path)

origin_point = extract_values_from_file(output_path, target_tag1, 3, 4)
convert_value = extract_values_from_file(output_path, target_tag2, 0, 1)

input_file = "./data/K3A_20190120043337_21100_00324917_L1G.txt"
output_file = './data/K3A_20190120043337_21100_00324917_L1G_trans.txt'
transform_values = origin_point

transform_and_save_wkt(input_file, output_file, transform_values, convert_value)

# WKT 파일의 좌표 체계 정보 추출
projection_info = extract_projection_info(geotiff_path)

# 원본 좌표 체계와 대상 좌표 체계 정의 (여기서는 WGS 84 좌표 체계를 대상으로 설정)
src_proj = projection_info
dst_proj = 'epsg:4326'

# 입력 파일과 출력 파일 지정
input_file = "./data/K3A_20190120043337_21100_00324917_L1G.txt"
final_output_file = "./data/K3A_20190120043337_21100_00324917_L1G_trans_final.txt"

# 좌표 변환 및 파일 저장
read_and_transform_linestring(output_file, final_output_file, src_proj, dst_proj)
"""