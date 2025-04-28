import matplotlib.pyplot as plt
import numpy as np

def generate_chessboard_pdf(board_size=(7, 10), square_size_mm=30, save_path='chessboard.pdf'):
    """
    生成棋盘格并保存为PDF，保证打印尺寸准确
    参数:
      board_size: (rows, cols) - 内角点数
      square_size_mm: 单格大小，单位mm
      save_path: 保存路径
    """
    rows = board_size[0] + 1  # 格子数 = 内角点数 + 1
    cols = board_size[1] + 1

    # 每格大小，单位m
    square_size_m = square_size_mm / 1000.0

    fig, ax = plt.subplots(figsize=(cols * square_size_m * 39.37, rows * square_size_m * 39.37))  # 39.37 inch/m
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制棋盘格
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                rect = plt.Rectangle((j, rows - i - 1), 1, 1, facecolor='black')
                ax.add_patch(rect)

    # 保存为PDF
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"棋盘格PDF已保存到：{save_path}")

# 示例
generate_chessboard_pdf(board_size=(7, 10), square_size_mm=10, save_path='chessboard_7x10_10mm.pdf')
