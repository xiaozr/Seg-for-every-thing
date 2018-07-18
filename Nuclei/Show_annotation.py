"""
show the GT mask(POLY format)
"""

import json
import os 
import pycocotools
from PIL import Image
from pycocotools import mask as mask_util
import numpy as np 
from pycocotools.coco import COCO


def get_mask(coco,img_meta):
	id = coco.getAnnIds(imgIds = img_meta )
	seg = coco.loadAnns(ids = id)
	mask_collection=[]
	for i in seg:
		mid = coco.annToRLE(i)  # input: ann 
		mask_collection.append(mid)
	if len(mask_collection) == 0:
		return None
	masks = mask_util.decode(mask_collection)
	return masks

def get_image(img_meta,coco):
	img = coco.loadImgs(img_meta)
	#print img
	img_name = img[0]['file_name']
	return img_name

def blend_two_images(img1, img2, alpha=0.5):
	img1 = img1.convert('RGBA')
	img2 = img2.convert('RGBA')
	img = Image.blend(img1, img2, alpha) # img = img1 * (1 - alpha) + img2 * alpha
	return img 

def main():
	coco = COCO(anno_dir)
	#get all image_id in minval
	all_image_Id = coco.getImgIds()
	for j in all_image_Id:
		print('process the img_id = {}'.format(j))
		#get all mask in one image
		mask_j = get_mask(coco,j)
		#get one image
		img_name = get_image(j,coco)
		#no mask if score under threshold
		if mask_j is None:
			img = Image.open(img_dir+img_name)
			img.save(save_dir+img_name,mode='RGBA')
		else:
			print('mask_j shape is ',mask_j.shape)
			mask_img_collection=[]
			for i in range(mask_j.shape[2]):
				mask_j_ = mask_j[:,:,i]
				c = (np.random.random((1,3))*0.6+0.4).tolist()[0]
				mid = np.zeros((mask_j.shape[0],mask_j.shape[1],3))
				#print(mid.shape,mask_j_.shape)
				mid[:,:,0] = mask_j_*c[0]*255
				mid[:,:,1] = mask_j_*c[1]*255 
				mid[:,:,2] = mask_j_*c[2]*255
				mask_img_collection.append(mid)
			# add all mask to one image	
			mask = np.zeros((mask_j.shape[0],mask_j.shape[1],3))
			for i in mask_img_collection:
				mask =mask +i
			mask = np.asarray(mask,dtype=np.uint8)
			mask = Image.fromarray(mask)
			img = Image.open(img_dir+img_name)
			#blend mask and image
			final_image = blend_two_images(img,mask)
			final_image.save(save_dir+img_name,mode='RGBA')

if __name__ == '__main__':
	curr_dir = os.getcwd()
	result_dir = '/data1/shuai/Nuclei/annotations/test_stage_1_local_train_split.json'
	dic = json.load(open(result_dir))
	anno_dir = '/data1/shuai/Nuclei/annotations/test_stage_1_local_train_split.json'
	img_dir = '/data1/shuai/Nuclei/stage_1_train/'
	save_dir = '/data1/shuai/Nuclei/output/'
	main()
