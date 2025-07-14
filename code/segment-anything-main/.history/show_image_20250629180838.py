import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_coordinates():
    # 读取图像
    image = cv2.imread(r'D:\workstation\sam\segment-anything-main\image\block.png')
    if image is None:
        print("错误：无法读取图像文件")
        return
    
    # 转换颜色空间从BGR到RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    print(f"Image size: width={w}, height={h}")
    
    # 创建图像窗口
    plt.figure(figsize=(15, 10))
    
    # 显示图像
    plt.imshow(image)
    
    # 添加主网格线（每200像素）
    plt.grid(True, color='white', alpha=0.8, linewidth=0.8)
    
    # 添加次网格线（每50像素）
    plt.grid(True, which='minor', color='white', alpha=0.2, linewidth=0.5)
    
    # 设置主刻度（每200像素）
    major_ticks_x = np.arange(0, w, 200)
    major_ticks_y = np.arange(0, h, 200)
    plt.gca().set_xticks(major_ticks_x)
    plt.gca().set_yticks(major_ticks_y)
    
    # 设置次刻度（每50像素）
    minor_ticks_x = np.arange(0, w, 50)
    minor_ticks_y = np.arange(0, h, 50)
    plt.gca().set_xticks(minor_ticks_x, minor=True)
    plt.gca().set_yticks(minor_ticks_y, minor=True)
    
    # 添加标题和标签
    plt.title("original image with coordinates")
    plt.xlabel("X ")
    plt.ylabel("Y ")
    
    # 启用坐标显示
    def format_coord(x, y):
        x, y = int(x), int(y)
        if 0 <= x < w and 0 <= y < h:
            return f'x={x}, y={y}'
        return ''
    
    plt.gca().format_coord = format_coord
    
    # 显示鼠标位置提示
    print("\nUsage:")
    print("- Move mouse over image to see coordinates")
    print("- Major grid lines: every 200 pixels (solid white)")
    print("- Minor grid lines: every 50 pixels (faint white)")
    print("- Click on image to read coordinates")
    print(f"- Valid coordinate ranges: x: 0-{w-1}, y: 0-{h-1}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    show_image_with_coordinates()
