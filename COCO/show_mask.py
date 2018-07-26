import json
import os 
import pycocotools
from PIL import Image
from pycocotools import mask as mask_util
import numpy as np 
from pycocotools.coco import COCO
import cv2
"""
draw the coco format output
"""

def get_mask(img_meta,threshold=0.9):
 """
 image_meta: the image id 
 threshold: threshold of score
 return: array of masks
 """
 image_annotations=[]
 for i in dic:
  if i['image_id'] == img_meta and i['score'] > threshold: 
   image_annotations.append(i['segmentation'])
   #print i['segmentation']
 if len(image_annotations) == 0:
  return None
 masks = mask_util.decode(image_annotations)
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

def blend_masks_in_one_image(masks,img_name,img_dir,show_border=True,border_thick=1):
 """
  masks: array of mask 
  img_name: the image name 
  img_dir: the dir of image
  show_border: show the border of mask 
  border_thick: the thick of border
  return: PIL image
 """
 #no mask if score under threshold
 img = Image.open(img_dir+img_name)
 if masks is None:
  return img
 else:
  print('masks shape is ',masks.shape)
  mask_img_collection=[]
  for i in range(masks.shape[2]):
   masks_ = masks[:,:,i]
   c = (np.random.random((1,3))*0.6+0.4).tolist()[0]
   mid = np.zeros((masks.shape[0],masks.shape[1],3))
   #print(mid.shape,masks_.shape)
   mid[:,:,0] = masks_*c[0]*255
   mid[:,:,1] = masks_*c[1]*255 
   mid[:,:,2] = masks_*c[2]*255
   mask_img_collection.append(mid)
   if show_border:
    img = np.asarray(img)
    mask_new,contours,hierarchy = cv2.findContours(
     masks_.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
          cv2.drawContours(img, contours, -1, (255,255,255), border_thick, cv2.LINE_AA)
          img = Image.fromarray(img)
  # add all mask to one image 
  mask = np.zeros((masks.shape[0],masks.shape[1],3))
  for i in mask_img_collection:
   mask =mask +i
  mask = np.asarray(mask,dtype=np.uint8)
  mask = Image.fromarray(mask)
  #blend mask and image
  img = blend_two_images(img,mask)
  return img

def draw_bbox_and_class(coco,img_name,image,image_id,threshold=0.9,thick=1,show_class_and_score=True,font_scale=0.35):
 """
 image: PIL image
 image id: the image id 
 return: cv2 image
 """
 image = np.asarray(image)
 #1) pred box
 bbox_file = json.load(open(pred_bbox_json))
 for i in bbox_file:
  if i['image_id'] == image_id and i['score'] > threshold :
   #1)box
   #left_top
   vertex_1 = int(i['bbox'][0])
   vertex_2 = int(i['bbox'][1])
   #right_down
   vertex_3 = vertex_1 + int(i['bbox'][2])
   vertex_4 = vertex_2 + int(i['bbox'][3])
   #red for pred
   image = cv2.rectangle(image,(vertex_1,vertex_2),(vertex_3,vertex_4),color=(0,0,255),thickness=thick)
   #2)show class
   if show_class_and_score:
    class_name = coco.loadCats(i['category_id'])
    class_name = class_name[0]['name']
    txt_str = class_name + ' score ' + '{:0.2f}'.format(i['score'])
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w,txt_h),_) = cv2.getTextSize(txt_str, font, font_scale, 1)
    # Place text background.
    back_tl = vertex_1, vertex_2 - int(1.3 * txt_h)
    back_br = vertex_1 + txt_w, vertex_2
    image = cv2.rectangle(image, back_tl, back_br, (18, 127, 15), -1)
    # Show text.
    txt_tl = vertex_1, vertex_2 - int(0.3 * txt_h)
    image = cv2.putText(image, txt_str, txt_tl, font, font_scale, (218, 227, 218), lineType=cv2.LINE_AA)
 #2) truth box
 truth_box_anno_id = coco.getAnnIds(imgIds=image_id)
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
  image = cv2.rectangle(image,(vertex_1,vertex_2),(vertex_3,vertex_4),color=(0,255,0),thickness=thick)

 #save result
 if not os.path.exists(save_dir):
  cmd = 'mkdir -p ' + save_dir
  os.system(cmd)
 cv2.imwrite(save_dir + img_name,image)


def main():
 coco = COCO(anno_dir)
 #get all image_id in minval
 all_image_Id = coco.getImgIds()
 for j in all_image_Id:
  print('process the img_id = {}'.format(j))
  #get all mask in one image
  mask_j = get_mask(j)
  #get one image
  img_name = get_image(j,coco)
  img = blend_masks_in_one_image(mask_j,img_name,img_dir,show_border=True,border_thick=1)
  draw_bbox_and_class(coco,img_name,img,j)
if __name__ == '__main__':
 curr_dir = os.getcwd()
 result_dir = '/data1/shuai/adas/detectron_output/test/adas_train_20171026/mask_rcnn/segmentations_adas_train_20171026_results.json'
 dic = json.load(open(result_dir))
 anno_dir = '/data1/shuai/adas/annotations/20171026.json'
 img_dir = '/data1/shuai/adas/20171026/'
 pred_bbox_json = '/data1/shuai/adas/detectron_output/test/adas_train_20171026/mask_rcnn/bbox_adas_train_20171026_results.json'
 save_dir = '/data1/shuai/adas/detectron_output/test/adas_train_20171026/mask_rcnn/mask_img/'
 main()
