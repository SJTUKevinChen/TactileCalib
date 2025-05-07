import cv2
import os
import numpy as np
import glob
from utils import *

# 棋盘格参数
pattern_size = (6, 4)  # 行交叉点数量 × 列交叉点数量
square_size = 4  # 单位 mm
aruco_size = 10  # 单位 mm

# 读取所有图像
output_dir = "./output"
images = glob.glob("./grid_images/*")
aruco_images = glob.glob("./origin_images/*")
val_images = glob.glob("./val/*")

# 对于一般图像获取内参
K, exts, dist = calibrateCamera(images, pattern_size, square_size, visualize=False)

print("Camera Intrinsic Matrix (K):\n", K)
print("Distortion Coefficients (dist):\n", dist)

# 利用原点图像网格，标定原点
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
Rt_avg = set_origin_aruco(aruco_images, K, dist, aruco_dict, 10, visualize=False)

# 保存K, Rt_avg, s_avg
param_path = os.path.join(output_dir, "camera_params.npz")
np.savez(param_path, K=K, Rt_avg=Rt_avg)

# 验证标定结果
pts = calib_val(val_images)

# 计算标定误差
for pt in pts:
    X,Y = get_coordinate_actual(pt["pixel"][0],pt["pixel"][1], param_path)
    err = np.linalg.norm(np.array(pt['real']) - np.array((X, Y))) / np.linalg.norm(np.array(pt['real']))
    print(f"真实坐标: {pt['real']}, 标定结果: ({X:.2f}, {Y:.2f}), Error: {err:.2%}")


