import json
import os 
import pycocotools
from PIL import Image
from pycocotools import mask as mask_util
import numpy as np 
from pycocotools.coco import COCO
import cv2

"""
check the converted annotation (maskxrcnn --> maskrcnn)
"""
def get_mask(coco,img_meta):
 id = coco.getAnnIds(imgIds = img_meta )
 seg = coco.loadAnns(ids = id)
 mask_collection=[]
 for i in seg:
  mid = coco.annToRLE(i)
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
 coco = COCO(result_dir)
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
   image = blend_two_images(img,mask)
   ##
   image = np.asarray(image)
   print(image.shape)
   #convert the RGBA to BGRA
   image.setflags(write=1)
   image[:,:,0:3] = image[:,:,-2::-1]
   truth_box_anno_id = coco.getAnnIds(imgIds=j)
   # reture is list of dic[{segmentation:,bbox:,,,},{,,,}]
   truth_box_anno = coco.loadAnns(truth_box_anno_id)
   for i in truth_box_anno:
    #left_top
    vertex_1 = int(i['bbox'][0])
    vertex_2 = int(i['bbox'][1])
    #right_down
    vertex_3 = vertex_1 + int(i['bbox'][2])
    vertex_4 = vertex_2 + int(i['bbox'][3])
    #green for pred
    image = cv2.rectangle(image,(vertex_1,vertex_2),(vertex_3,vertex_4),color=(0,255,0),thickness=1)
   ##
   cv2.imwrite(save_dir + img_name,image)

 main()
 
