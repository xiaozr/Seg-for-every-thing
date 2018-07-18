"""
Cityscapes
show the groundtruth mask and bbox , model's result of mask and bbox in one image
"""

from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np 
from os import path
import json
import cv2

def get_image(image_id,coco):
  """
  get image_name from image_id 
  """
  
	img = coco.loadImgs(image_id)
	#print img
	img_name = img[0]['file_name']
	return img_name

def blend_masks_in_one_image(image_name,result_dir,image_raw):
	"""
  threhold = 0.9
  blend masks( binary_png[0,255] ) in one PIL image([0-255])
  input:  image_name
          result_dir: dir of image of masks(model output) 
          image_raw: the image of the image_name
  return: image which collect all masks of the image
	"""
  
  mask_img_collection = []
	for _,_,i in os.walk(result_dir):
		for j in i:
			preix_name = j.split('_')
			preix_name = preix_name[0]+'_'+preix_name[1]+'_'+preix_name[2]+'_leftImg8bit'
			preix_img_name = image_name.split('.')[0]
			if preix_name == preix_img_name:
				#threshold 0.9
				pre_file = open(root_dir+preix_img_name+'pred.txt')
				for line in pre_file:
					line = line.split('/')[1]
					png = line.split(' ')[0]
					thre = line.split(' ')[2]
					if png == j and float(thre) > 0.9:
						c = (np.random.random((1,3))*0.6+0.4).tolist()[0]
						mid = Image.open(result_dir+j)
						mid = np.asarray(mid)
						mid_mask = np.zeros((mid.shape[0],mid.shape[1],3))
						mid_mask[:,:,0] = mid*c[0]
						mid_mask[:,:,1] = mid*c[1]
						mid_mask[:,:,2] = mid*c[2]
						mask_img_collection.append(mid_mask)
		if len(mask_img_collection) != 0:
			mask = np.zeros((mid.shape[0],mid.shape[1],3))
			print ('number of masks {}'.format(len(mask_img_collection)))
			for j in mask_img_collection:
				mask = mask + j
			mask = np.asarray(mask,dtype=np.uint8)
			image = Image.fromarray(mask)
		else:
			image = image_raw
	return image

def draw_pred_box(coco,image,image_id,threshold = 0.9):
	"""
  draw the groundtruth and model output bbox to image
	input:
        coco
        image: PIL image 
        image_id: the image id of the image
        threshold
  output:
        image: PIL image 
  """
  
	image = np.asarray(image,dtype=np.uint8)
	#1) predic box
	bbox_file = json.load(open(pred_bbox_json))
	for i in bbox_file:
		if i['image_id'] == image_id and i['score'] > threshold :
			#left_top
			vertex_1 = int(i['bbox'][0])
			vertex_2 = int(i['bbox'][1])
			#right_down
			vertex_3 = vertex_1 + int(i['bbox'][2])
			vertex_4 = vertex_2 + int(i['bbox'][3])
			#red for predic
			image = cv2.rectangle(image,(vertex_1,vertex_2),(vertex_3,vertex_4),color=(255,0,0))
	#2) truth box
	truth_box_anno_id = coco.getAnnIds(imgIds=image_id)
	truth_box_anno = coco.loadAnns(truth_box_anno_id)  # reture: list of dic[{segmentation:,bbox:,,,},{,,,}]
	for i in truth_box_anno:
		#left_top
		vertex_1 = int(i['bbox'][0])
		vertex_2 = int(i['bbox'][1])
		#right_down
		vertex_3 = vertex_1 + int(i['bbox'][2])
		vertex_4 = vertex_2 + int(i['bbox'][3])
		#green for truth
		image = cv2.rectangle(image,(vertex_1,vertex_2),(vertex_3,vertex_4),color=(0,255,0))

	image = Image.fromarray(image)
	return image


def blend_two_images(img1, img2, alpha=0.2):
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
		img_name = get_image(j,coco)
		image_raw = Image.open(img_dir + img_name)
		mask = blend_masks_in_one_image(img_name,result_dir,image_raw)
		final_image = blend_two_images(image_raw,mask)
		final_image = draw_pred_box(coco,final_image,j,threshold=0.9)
		if not path.exists(save_dir):
			cmd = 'mkdir -p ' + save_dir
			os.system(cmd)	
		final_image.save(save_dir+img_name,mode='RGBA')


if __name__ == '__main__':
	root_dir = '/data1/shuai/cityscapes/detectron_output_1/test/cityscapes_fine_instanceonly_seg_val/mask_rcnn/'
	anno_dir = '/home/wangshuai/seg_every_thing-master/lib/datasets/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
	img_dir = '/data1/shuai/cityscapes/images/'
	#path of mask rxcnn output
	result_dir = '/data1/shuai/cityscapes/detectron_output_1/test/cityscapes_fine_instanceonly_seg_val/mask_rcnn/results/'
	save_dir = '/data1/shuai/cityscapes/detectron_output_1/test/cityscapes_fine_instanceonly_seg_val/mask_rcnn/mask_img/'
	pred_bbox_json = '/data1/shuai/cityscapes/detectron_output_1/test/cityscapes_fine_instanceonly_seg_val/mask_rcnn/bbox_cityscapes_fine_instanceonly_seg_val_results.json'
	main()
