####
code for blend the output mask which belong to one image and the image
####
import json
import os 
import pycocotools
from PIL import Image
from pycocotools import mask as mask_util
import numpy as np 
from pycocotools.coco import COCO


def get_mask(img_meta):
	"""
  input: 
       img_meta:  one image_id
       dic: dictory of modle result
  return:
       masks:  the masks in one image
  """
  
  image_annotations=[]
	for i in dic:
    #only get the mask whose score > 0.9
		if i['image_id'] == img_meta and i['score'] > 0.9: 
			image_annotations.append(i['segmentation'])
			#print i['segmentation']
  #if one image have no mask ouput,return none    
	if len(image_annotations) == 0:
		return None
	masks = mask_util.decode(image_annotations)
	return masks

def get_image(img_meta,coco):
  """
  input:
       img_meta: one image_id
       coco: one coco object which open a json file
  ouput:
       img_name: image name of the image_id
  """
 
	img = coco.loadImgs(img_meta)
	img_name = img[0]['file_name']
	return img_name

def blend_two_images(img1, img2, alpha=0.5):
  """
  input:
        img1,img2: two PIL Image
        alpha:
  output:
        img: one blended PIL Image
  """
  
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
		mask_j = get_mask(j)
		#get the image name
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
	result_dir = '/data1/shuai/coco/detectron_output/test/coco_split_voc_2014_minival/mask_rcnn/segmentations_coco_split_voc_2014_minival_results.json'
	dic = json.load(open(result_dir))
	anno_dir = '/data1/shuai/coco/annotations/instances_minival2014.json'
	img_dir = '/data1/shuai/coco/images/coco_val2014/'
	save_dir = '/data1/shuai/coco/detectron_output/test//coco_split_voc_2014_minival/mask_rcnn/mask_img/'
	main()
