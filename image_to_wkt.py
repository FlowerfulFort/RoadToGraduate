import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from osgeo import gdal
from shapely import wkt
from itertools import tee
from other_tools import sknw
from functools import partial
from matplotlib.pylab import plt
from scipy import ndimage as ndi
from multiprocessing import Pool
from pyproj import Proj, Transformer
from shapely.wkt import loads, dumps
from shapely.affinity import translate
from shapely.geometry import Point, LineString
from collections import defaultdict, OrderedDict
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes

Image.MAX_IMAGE_PIXELS = None

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
args = parser.parse_args()

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_sequential_duplicates(seq):
    #todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx+1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx-1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_angle(p0, p1=np.array([0,0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def preprocess(img, thresh):
    img = (img > (255 * thresh)).astype(bool)
    remove_small_objects(img, 300)
    remove_small_holes(img, 300)
    # img = cv2.dilate(img.astype(np.uint8), np.ones((7, 7)))
    return img

def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(img, G, vertices):
    plt.imshow(img, cmap='gray')

    # draw edges by pts
    for (s, e) in G.edges():
        vals = flatten([[v] for v in G[s][e].values()])
        for val in vals:
            ps = val.get('pts', [])
            plt.plot(ps[:, 1], ps[:, 0], 'green')

    # draw node by o
    node, nodes = G.node(), G.nodes
    # deg = G.degree
    # ps = np.array([node[i]['o'] for i in nodes])
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], 'r.')

    # title and show
    plt.title('Build Graph')
    plt.show()

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

def remove_small_terminal(G):
    deg = G.degree()
    terminal_points = [i for (i, d) in deg if d == 1]
    edges = list(G.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            vals = flatten([[v] for v in G[s][s].values()])
            for ix, val in enumerate(vals):
                sum_len += len(val['pts'])
            if sum_len < 3:
                G.remove_edge(s, e)
                continue
        vals = flatten([[v] for v in G[s][e].values()])
        for ix, val in enumerate(vals):
            if s in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(s)
            if e in terminal_points and val.get('weight', 0) < 10:
                G.remove_node(e)
    return

linestring = "LINESTRING {}"

def make_skeleton(root, fn, debug, threshes, fix_borders):
    replicate = 5
    clip = 2
    rec = replicate + clip
    # open and skeletonize
    # img = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(os.path.join(root), cv2.IMREAD_GRAYSCALE)
    
    if fix_borders:
        img = cv2.copyMakeBorder(img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE)
    img_copy = None
    if debug:
        if fix_borders:
            img_copy = np.copy(img[replicate:-replicate,replicate:-replicate])
        else:
            img_copy = np.copy(img)
   
    thresh = threshes['2']
    img = preprocess(img, thresh)
    if not np.any(img):
        return None, None
    ske = skeletonize(img).astype(np.uint16)
    if fix_borders:
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0)
    return img_copy, ske


def add_small_segments(G, terminal_points, terminal_lines):
    node = G.nodes()
    term = [node[t]['o'] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < 20))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > 20) & (dists < 100))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > 20:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if -20 < angle < 20 or angle < -160 or angle > 160:
            good_pairs.append((s, e))

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'], G.nodes[e]['o']
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
    return wkt


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def build_graph(root, fn, debug=False, threshes={'2': .3, '3': .3, '4': .3, '5': .2}, add_small=True, fix_borders=True):
    city = os.path.splitext(fn)[0]

    img_copy, ske = make_skeleton(root, fn, debug, threshes, fix_borders)
    if ske is None:
        return city, [linestring.format("EMPTY")]
    G = sknw.build_sknw(ske, multi=True)
    remove_small_terminal(G)
    node_lines = graph2lines(G)
    if not node_lines:
        return city, [linestring.format("EMPTY")]
    node = G.nodes()
    deg = G.degree()
    wkt = []
    terminal_points = [i for i, d in deg if d == 1]

    terminal_lines = {}
    vertices = []
    for w in node_lines:
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            vals = flatten([[v] for v in G[s][e].values()])
            for ix, val in enumerate(vals):

                s_coord, e_coord = node[s]['o'], node[e]['o']
                pts = val.get('pts', [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        wkt.extend(add_small_segments(G, terminal_points, terminal_lines))

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return city, [linestring.format("EMPTY")]
    return city, wkt

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

def convert_coordinate(src_proj, dst_proj, coordinates):
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
                    [Point(convert_coordinate(src_proj, dst_proj, coords)) for coords in linestring.coords]
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

# 좌표 변환
def convert_coordinates(image_path, intermediate_dir, intermediate1, final_output, image_name):
    tags_file = os.path.join(intermediate_dir, f"{image_name}_tags.txt")
    _ = extract_tags_from_tif(image_path, tags_file)
    intermediate2 = os.path.join(intermediate_dir, f"{image_name}_intermediate2.txt")
    target_tag1, target_tag2 = 33922, 33550

    origin_point = extract_values_from_file(tags_file, target_tag1, 3, 4)
    convert_value = extract_values_from_file(tags_file, target_tag2, 0, 1)
    
    transform_and_save_wkt(intermediate1, intermediate2, origin_point, convert_value)

    projection_info = extract_projection_info(image_path)

    src_proj, dst_proj = projection_info, 'epsg:4326'

    read_and_transform_linestring(intermediate2, final_output, src_proj, dst_proj)
            
# 이미지를 WKT로 변환
def convert_image_to_wkt(image_path, intermediate_dir, final_output, image_name):
    target_path = os.path.join(intermediate_dir, f"{image_name}_mask.png")
    txt_name = os.path.join(intermediate_dir, f"{image_name}_intermediate1.txt")

    with Pool() as p:
        data = p.map(partial(build_graph, target_path), [target_path])

    all_data = [val for _, v in data for val in v]

    with open(txt_name, 'w') as file:
        file.write("\n".join(all_data))
        
    convert_coordinates(image_path, intermediate_dir, txt_name, final_output, image_name)


if __name__ == "__main__":
    prefix = ''
    results_root = args.image_path
    txt_name = os.path.join(os.path.dirname(os.path.abspath(args.image_path)), args.image_path.split('/')[-1].split('.')[0] + ".txt")
    root = os.path.join(results_root)
    f = partial(build_graph, root)
    #l = [v for v in os.listdir(root) if prefix in v]
    #l = list(sorted(l))
    l = [root]
    with Pool() as p:
        data = p.map(f, l)
    all_data = []
    for _, v in data:
        for val in v:
            all_data.append(val)
            
    with open(txt_name, 'w') as file:
        for line in all_data:
            file.write(line + "\n")
    #df = pd.DataFrame(all_data, columns=['WKT_Pix'])
    #df.to_csv(txt_name, index=False)