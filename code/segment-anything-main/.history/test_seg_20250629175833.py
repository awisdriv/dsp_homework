import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
#dog 1320 800       750 
# 设置matplotlib后端
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端

def show_mask(mask, ax, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		color = np.array([30/255, 144/255, 255/255, 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)
		    
def show_points(coords, labels, ax, marker_size=375):
	pos_points = coords[labels==1]
	neg_points = coords[labels==0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
		    
def show_box(box, ax):
	x0, y0 = box[0], box[1]
	w, h = box[2] - box[0], box[3] - box[1]
	ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# 读取图像
image = cv2.imread(r'D:\workstation\sam\segment-anything-main\image\dog.png')
if image is None:
	print("错误：无法读取图像文件")
	exit()
print(f"图像尺寸: {image.shape}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# 加载SAM模型
sam_checkpoint = r"D:\workstation\sam\segment-anything-main\sam_vit_h_4b8939.pth"
model_type = "vit_h" 
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 设置图像
predictor.set_image(image)

# 获取图像尺寸
h, w = image.shape[:2]
print(f"图像高度: {h}, 宽度: {w}")

# 获取用户输入的点坐标
print("\n请输入点的坐标（x y）:")
point_x, point_y = map(int, input().split())
input_points = np.array([[point_x, point_y]])
input_labels = np.array([1])

# 获取用户输入的框坐标
print("\n请输入框的坐标（左上角x 左上角y 右下角x 右下角y）:")
box_x1, box_y1, box_x2, box_y2 = map(int, input().split())
input_box = np.array([box_x1, box_y1, box_x2, box_y2])

# 创建图像显示窗口（2行1列的子图）
plt.figure(figsize=(10, 20))

# 第一个子图：使用点提示
ax1 = plt.subplot(2, 1, 1)
ax1.imshow(image)

# 使用点提示进行预测
masks_points, scores_points, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# 选择得分最高的mask
best_mask_idx = np.argmax(scores_points)
best_mask_points = masks_points[best_mask_idx]

# 显示点提示结果
show_mask(best_mask_points, ax1)
show_points(input_points, input_labels, ax1)
ax1.set_title(f"point (score: {scores_points[best_mask_idx]:.3f})")
ax1.axis('on')

# 第二个子图：使用框提示
ax2 = plt.subplot(2, 1, 2)
ax2.imshow(image)

# 使用框提示进行预测
masks_box, scores_box, logits = predictor.predict(
    box=input_box[None, :],
    multimask_output=True,
)

# 选择得分最高的mask
best_mask_idx = np.argmax(scores_box)
best_mask_box = masks_box[best_mask_idx]

# 显示框提示结果
show_mask(best_mask_box, ax2)
show_box(input_box, ax2)
ax2.set_title(f"box (score: {scores_box[best_mask_idx]:.3f})")
ax2.axis('on')

# 调整子图之间的间距
plt.tight_layout()

# 显示结果
print("正在显示分割结果...")
plt.show(block=True)

# 清理
plt.close('all')

		
