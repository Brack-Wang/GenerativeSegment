import numpy as np
import tifffile as tiff
import os
from skimage.morphology import skeletonize_3d
import networkx as nx

def read_tiff_3d(file_path):
    """ 读取 3D TIFF 文件并转换为 NumPy 数组 """
    volume = tiff.imread(file_path)  # 读取 3D mask
    volume = (volume > 0).astype(np.uint8)  # 转换为二值
    return volume

def extract_skeleton(volume):
    """ 进行 3D 骨架提取 """
    skeleton = skeletonize_3d(volume)
    return skeleton

def skeleton_to_swc(skeleton, output_swc):
    """ 将 3D 骨架转换为 SWC 格式并保存 """
    indices = np.argwhere(skeleton > 0)  # 获取骨架体素坐标
    G = nx.Graph()  # 生成无向图（用于计算连接性）

    # 创建 SWC 节点
    swc_data = []
    node_id = 1
    node_map = {}  # 体素坐标到 SWC ID 映射

    for i, (z, y, x) in enumerate(indices):
        node_map[(x, y, z)] = node_id
        swc_data.append([node_id, 2, x, y, z, 1.0, -1])  # 默认半径 1.0，根节点 -1
        node_id += 1

    # 计算连接关系
    for (x, y, z) in node_map.keys():
        for dx, dy, dz in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in node_map:
                G.add_edge(node_map[(x, y, z)], node_map[neighbor])

    # 选择 SWC 根节点（默认选择最顶部的点）
    root_id = min(node_map.values(), key=lambda i: swc_data[i-1][3])  # 选取 Y 坐标最小的作为根节点
    for u, v in G.edges:
        if u != root_id:
            swc_data[u - 1][-1] = v  # 连接到最近的骨架点

    # 保存为 SWC 格式
    with open(output_swc, 'w') as f:
        f.write("# SWC file generated from 3D skeletonization\n")
        f.write("# id type x y z radius parent\n")
        for line in swc_data:
            f.write(" ".join(map(str, line)) + "\n")

def process_3d_tiff_to_swc(input_tiff, output_swc):
    """ 读取 TIFF -> 提取骨架 -> 保存为 SWC """
    print(f"Reading 3D TIFF: {input_tiff}")
    volume = read_tiff_3d(input_tiff)

    print("Extracting 3D skeleton...")
    skeleton = extract_skeleton(volume)

    print(f"Saving SWC file: {output_swc}")
    skeleton_to_swc(skeleton, output_swc)
    print("Processing complete!")

# 示例：运行代码（请修改路径）
input_tiff_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/large/filtered.tif"  # 你的 3D TIFF 文件路径
output_swc_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/morf3/large/output_ndl.swc"  # SWC 输出路径

process_3d_tiff_to_swc(input_tiff_path, output_swc_path)
