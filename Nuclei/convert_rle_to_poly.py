"""
change the compact rle annotation to poly
"""

import json 
from pycocotools import mask as mask_util
import cv2
import numpy as np 

def main():
	raw = json.load(open(raw_file))
	raw_annotations = raw['annotations']
	j = 0
	for i in raw_annotations:
		seg = i['segmentation']
		mask = mask_util.decode(seg)
		mask_ = mask.astype(np.uint8).copy() 
		mask_new,contours,hierarchy = cv2.findContours(mask_,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		segmentation = []
		for contour in contours:
			contour = contour.flatten().tolist()
			if len(contour) >4:
				segmentation.append(contour)
		raw['annotations'][j]['segmentation'] = segmentation
		j = j+1 
		print i['id']
	
	with open(dst_file,'w') as f:
		json.dump(raw,f)

if __name__ == '__main__':
	raw_file = '/data1/shuai/Nuclei/annotations/stage_1_local_val_split.json'
	dst_file = '/data1/shuai/Nuclei/annotations/test_stage_1_local_val_split.json'
	main()
