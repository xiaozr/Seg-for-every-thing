import json
import pycocotools
from pycocotools import mask as mask_util 
from pycocotools.coco import COCO
import cv2
import numpy as np 
import os 
'''
convert the detectron output segment annotation to the citysapce's type 
a gary image to save the instanceID
save the first five clase and the last four class separatly 
for first five:
  car:[101,130]
  bus:[131,160]
  truck:[161,190]
  person:[191,220]
  non_motro_vehicle:[221,250]
for last four:
  road_sign:[101,130]
  traffic_sign:[131,160]
  car_side_wheel:[161,190]
  traffic_light_idx:[191,221]
'''
def get_mask(img_meta,seg_dic,threshold=0.9):
 """
 image_meta: the image id 
 threshold: threshold of score
 return: array of masks
 """
 image_annotations= []
 categories = []
 for i in seg_dic:
  if i['image_id'] == img_meta and i['score'] > threshold: 
   image_annotations.append(i['segmentation'])
   #print i['segmentation']
   categories.append(i['category_id'])
 if len(image_annotations) == 0:
  return [],[]
 masks = mask_util.decode(image_annotations)
 return masks,categories

def get_image(img_meta,coco):
 img = coco.loadImgs(img_meta)
 #print img
 img_name = img[0]['file_name']
 return img_name
def draw_img_last_four(masks,categories,img_name):
 #print(masks.shape,'maskshape')
 img  = np.zeros([720,1280])
 Road_sign_idx = 101
 traffic_sign_idx = 131
 car_side_wheel_idx = 161
 traffic_light_idx = 191
 for i in range(len(categories)):
  #print('category',categories[i])
  if categories[i] == 6:
   img[np.where(masks[:,:,i]!=0)] = Road_sign_idx
   Road_sign_idx = Road_sign_idx + 1
  if categories[i] == 7:
   img[np.where(masks[:,:,i]!=0)] = traffic_sign_idx
   traffic_sign_idx = traffic_sign_idx + 1
  if categories[i] == 9:
   img[np.where(masks[:,:,i]!=0)] = traffic_light_idx
   traffic_light_idx = traffic_light_idx + 1
  if categories[i] == 8:
   img[np.where(masks[:,:,i]!=0)] = car_side_wheel_idx
   car_side_wheel_idx = car_side_wheel_idx + 1 
 if not os.path.exists(save_dir_last):
  cmd = 'mkdir -p ' + save_dir_last
  os.system(cmd)
 cv2.imwrite(save_dir_last+img_name,img)
def draw_img_first_five(masks,categories,img_name):
 #print(masks.shape,'maskshape')
 img  = np.zeros([720,1280])
 car_idx = 101
 bus_idx = 131
 truck_idx = 161
 person_idx = 191
 non_motor_vehicle_idx = 221
 for i in range(len(categories)):
  if categories[i] == 1:
   img[np.where(masks[:,:,i]!=0)] = car_idx
   car_idx = car_idx + 1
  if categories[i] == 2:
   img[np.where(masks[:,:,i]!=0)] = bus_idx
   bus_idx = bus_idx + 1
  if categories[i] == 3:
   img[np.where(masks[:,:,i]!=0)] = truck_idx
   truck_idx = truck_idx + 1
  if categories[i] == 4:
   img[np.where(masks[:,:,i]!=0)] = person_idx
   person_idx = person_idx + 1
  if categories[i] == 5:
   img[np.where(masks[:,:,i]!=0)] = non_motor_vehicle_idx
   non_motor_vehicle_idx = non_motor_vehicle_idx + 1
 if not os.path.exists(save_dir_first):
  cmd = 'mkdir -p ' + save_dir_first
  os.system(cmd)
 cv2.imwrite(save_dir_first+img_name,img)
def main():
 coco = COCO(anno_dir)
 all_image_id = coco.getImgIds()
 seg_dic = json.load(open(seg_file))
 for j in all_image_id:
  print('process the img_id = {}'.format(j))
  mask_j,category_j = get_mask(j,seg_dic)
  img_name = get_image(j,coco)
  draw_img_first_five(mask_j,category_j,img_name)
  draw_img_last_four(mask_j,category_j,img_name)  

if __name__ == '__main__':
 seg_file = '/data1/shuai/adas/day_night_output/test/adas_train_day/mask_rcnn/segmentations_adas_train_day_results.json'
 anno_dir = '/data1/shuai/adas/annotations/img_day_train_22000.json'
 save_dir_first = '/data1/shuai/adas/day_night_output/test/adas_train_day/mask_rcnn/cityscapes_type_anno_5/'
 save_dir_last = '/data1/shuai/adas/day_night_output/test/adas_train_day/mask_rcnn/cityscapes_type_anno_4/'
 main()
