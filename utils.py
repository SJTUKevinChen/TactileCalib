import cv2
import cv2.aruco as aruco
import numpy as np

def get_ext_matrix(rvecs, tvecs):
    """
    输入:
      rvecs - 旋转向量列表
      tvecs - 平移向量列表
    输出:
      ext_matrices - 外参矩阵列表 (4x4)
    """
    ext_matrices = []

    for rvec, tvec in zip(rvecs, tvecs):
        # 1. 将旋转向量转换成旋转矩阵
        R, _ = cv2.Rodrigues(rvec)  # R 是 3x3 旋转矩阵

        # 2. 把 tvec 转成 3x1 列向量
        t = tvec.reshape(3, 1)

        # 3. 构造 4x4 的齐次变换矩阵 T = [R | t]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        ext_matrices.append(T)

    return ext_matrices

def calibrateCamera(images, pattern_size, square_size, visualize=False):
    """
    输入:
      images - 图像列表
      pattern_size - 棋盘格的行列数 (rows, cols)
      square_size - 棋盘格每个方格的大小，单位 mm
    输出:
      K - 相机内参矩阵 (3x3)
      exts - 外参矩阵列表 (4x4)
      dist - 畸变系数
      visualize - 是否可视化角点
    """
    objpoints = []  # 3d 点在世界坐标系中的位置
    imgpoints = []  # 2d 点在图像平面中的位置

    # 世界坐标中的棋盘格点
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    objp *= square_size

    # 读取所有图像
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("img", gray)
        # cv2.waitKey(0)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        
        if ret:
            print(f"角点检测成功: {fname}")
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"角点检测失败: {fname}")
    
        # 显示角点
        if ret and visualize:
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow("Corners", cv2.resize(img,(1280, 960)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    # 校验是否有角点成功检测
    if len(objpoints) == 0:
        raise ValueError("没有成功检测到任何棋盘格角点，请检查图像质量或 pattern_size 设置是否正确")

    # 相机标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 计算各个外参
    exts = get_ext_matrix(rvecs, tvecs)

    return K, exts, dist

def average_Rt(rvecs, tvecs):
    """
    计算平均的外参矩阵
    输入:
      rvecs - 旋转向量列表
      tvecs - 平移向量列表
    输出:
      T_avg - 平均外参矩阵 (3x4：[R,t])
    """
    # 计算平均平移向量
    t_avg = np.mean(tvecs, axis=0).reshape(3, 1)

    # 计算平均旋转矩阵
    M = np.zeros((3, 3))
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        M += R

    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        # 保证是一个右手系
        R_avg = U @ np.diag([1, 1, -1]) @ Vt

    # 组合外参矩阵
    Rt = np.hstack((R_avg, t_avg))  # 3x4矩阵

    return Rt

def set_origin_aruco(images, K, dist, aruco_dict, marker_length_mm, visualize=False):
    """
    输入:
      images - ArUco 图像列表
      K - 相机内参矩阵 (3x3)
      dist - 畸变系数
      aruco_dict - ArUco 字典
      marker_length_mm - ArUco 标记的边长，单位 mm
    输出:
      Rt_avg - 平均外参矩阵 (3x4)
    """
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    rvecs_collect = []
    tvecs_collect = []

    # 预定义Aruco的四个角的真实世界坐标（以中心为原点），单位 mm
    half_size = marker_length_mm / 2.0
    obj_corners = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ])  # 形状 (4, 3)，单位 mm

    for fname in images:
        print(f"正在处理图像: {fname}")
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # 可视化标记
        output = aruco.drawDetectedMarkers(image.copy(), corners, ids)

        # 若检测到，则估计姿态并绘制坐标轴
        if ids is not None:
            print(f"检测到的 ArUco 角点坐标:\n {corners}")
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_mm, K, dist)

            img_corners = corners[0][0]  # (4,2)

            rvecs_collect.append(rvec)
            tvecs_collect.append(tvec)

            if visualize:
              # 绘制坐标系轴，长度单位为 mm
              output = cv2.drawFrameAxes(output, K, dist, rvec, tvec, marker_length_mm * 0.5)

              cv2.imshow("ArUco Coordinate Axis", cv2.resize(output,(1280, 960)))
              cv2.waitKey(0)
              cv2.destroyAllWindows()
        else:
            print(f"未检测到 ArUco 标记: {fname}")
            continue
        
    if len(rvecs_collect) == 0:
        raise ValueError("没有成功检测到任何 ArUco 标记，请检查图像质量或 marker_length_mm 设置是否正确")

    # 统一求平均
    Rt_avg = average_Rt(rvecs_collect, tvecs_collect)

    return Rt_avg

def get_coordinate_actual(u, v, param_path):
    """
    输入: 
      u, v - 像素坐标
      param_path - 标定参数文件路径
    输出:
      X, Y - 在Aruco平面坐标系下的坐标
    """
    # 读取标定参数
    param = np.load(param_path)
    K = param['K']  # 内参矩阵
    Rt = param['Rt_avg']  # 外参矩阵

    # 计算方程Ax=b最小二乘解
    # 输入像素 (齐次)
    uv1 = np.array([u, v, 1.0]).reshape(3, 1)
    # 只取R的前两列 + t列 (3x3矩阵)
    T = Rt[:, [0,1,3]]
    A_full = (uv1 @ np.array([0, 0, 1]).reshape(1, 3) - K) @ T
    A1A2 = A_full[:,[0,1]]
    b = -A_full[:,2].reshape(3, 1)

    # 求解最小二乘
    xy = np.linalg.inv(A1A2.T @ A1A2) @ A1A2.T @ b
    print(f"xy: {xy}")

    return xy[0,0], xy[1,0]

def calib_val(images):
    """
    鼠标点击获取一个点坐标的并可视化,取最后一次的点击点,键盘输入y确认坐标
    输入:
      image - 图像
    输出:
      pts - 点击的点坐标和真实物理坐标字典列表{'real': (x, y), 'pixel': (u, v)}
    """
    pts = []
    
    for img in images:
        
        print("请点击图像获取点坐标，按 'y' 键确认选择的点")

        pt = {}
        pt_tmp = None

        # 从文件名中读取真实坐标:(mm, mm),文件名'(image_x_y.jpg)'
        filename = img.split('/')[-1]
        print(f"正在处理图像: {filename}")

        real_x, real_y = map(float, filename.split('.')[0].split('_')[1:])
        pt['real'] = (real_x, real_y)

        image = cv2.imread(img)

        def click_event(event, x, y, flags, param):
            nonlocal pt_tmp
            if event == cv2.EVENT_LBUTTONDOWN:
                pt_tmp = (x, y)
                print(f"点击坐标: {pt_tmp}")

                image_copy = image.copy()
                cv2.circle(image_copy, pt_tmp, 5, (0, 255, 0), -1)
                cv2.imshow("Image", image_copy)

        cv2.imshow("Image", image)
        cv2.setMouseCallback("Image", click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                break
        
        if pt_tmp is not None:
            pt['pixel'] = pt_tmp
            pts.append(pt)
            print(f"确认坐标: {pt_tmp}")
        else:
            print("没有选择坐标，跳过此图像")

    cv2.destroyAllWindows()
    return pts
    
    