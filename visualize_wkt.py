from shapely import wkt
from shapely.wkt import loads
import matplotlib.pyplot as plt

def visualize_wkt(wkt_data, output_file):
    fig, ax = plt.subplots(figsize=(304, 303))
    
    for wkt_string in wkt_data:
        geometry = loads(wkt_string)
        x, y = geometry.xy
        ax.plot(x, y, linestyle='solid', linewidth=30) 
        
    ax.axis('off')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    plt.savefig(output_file, format='png')
    plt.show()

# 파일에서 WKT 데이터를 읽기
file_path = './data/K3A_20190120043337_21100_00324917_L1G.txt'  # 파일 경로를 해당 파일의 실제 경로로 변경해야 합니다.
with open(file_path, 'r') as file:
    wkt_data = file.readlines()
    
#wkt_data = ['LINESTRING (30 10, 10 30, 40 40)']

visualize_wkt(wkt_data, 'K3A_20190120043337_21100_00324917_L1G_plt.png')