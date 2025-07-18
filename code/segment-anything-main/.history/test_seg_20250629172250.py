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


image = cv2.imread(r'D:\workstation\sam\segment-anything-main\image\dog.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
		
sam_checkpoint = r"D:\workstation\sam\segment-anything-main\sam_vit_h_4b8939.pth"
model_type = "vit_b"
		
device = "cuda"  # or  ""
		
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
		
predictor = SamPredictor(sam)
# 单点 prompt  输入格式为(x, y)和并表示出点所带有的标签1(前景点)或0(背景点)。
input_point = np.array([[270, 240]])  # 标记点
input_label = np.array([1])  # 点所对应的标签
		
plt.figure(figsize=(10,10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()
masks, scores, logits = predictor.predict(
	    point_coords=input_point,
	    point_labels=input_label,
	    multimask_output=True,
	)
print(masks.shape)  # (number_of_masks) x H x W
		

input_point = np.array([[1308, 847], [1309, 845]])
input_label = np.array([1, 1])
		
mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
		
masks, _, _ = predictor.predict(
	point_coords=input_point,
	point_labels=input_label,
	mask_input=mask_input[None, :, :],
	multimask_output=False,)
		
print(masks.shape)
		
plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

		
