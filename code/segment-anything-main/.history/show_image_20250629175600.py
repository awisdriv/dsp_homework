import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_coordinates():
    # 读取图像
    image = cv2.imread(r'D:\workstation\sam\segment-anything-main\image\dog.png')
    if image is None:
        print("错误：无法读取图像文件")
        return
    
    # 转换颜色空间从BGR到RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    print(f"图像尺寸: 宽度={w}, 高度={h}")
    
    # 创建图像窗口
    plt.figure(figsize=(12, 8))
    
    # 显示图像
    plt.imshow(image)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴刻度
    plt.xticks(np.arange(0, w, 100))
    plt.yticks(np.arange(0, h, 100))
    
    # 添加标题和标签
    plt.title("original image with coordinates")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    
    # 启用坐标显示
    def format_coord(x, y):
        return f'x={int(x)}, y={int(y)}'
    plt.gca().format_coord = format_coord
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    show_image_with_coordinates()
