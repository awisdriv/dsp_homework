import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
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
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# 加载SAM模型
sam_checkpoint = r"D:\workstation\sam\segment-anything-main\sam_vit_h_4b8939.pth"
model_type = "vit_b"
device = "cuda"  # or "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 设置图像
predictor.set_image(image)

# 定义点提示和框提示
input_points = np.array([[500, 375]])  # 在狗的身体上的点
input_labels = np.array([1])  # 1表示前景点

# 定义包围狗的框
input_box = np.array([300, 200, 700, 550])  # [x1, y1, x2, y2]格式

# 使用点提示和框提示进行预测
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    box=input_box[None, :],  # 添加一个维度以匹配预期格式
    multimask_output=True,
)

# 选择得分最高的mask
best_mask_idx = np.argmax(scores)
best_mask = masks[best_mask_idx]

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(best_mask, plt.gca())
show_points(input_points, input_labels, plt.gca())
show_box(input_box, plt.gca())
plt.title(f"Mask Score: {scores[best_mask_idx]:.3f}")
plt.axis('off')
plt.show()

		
