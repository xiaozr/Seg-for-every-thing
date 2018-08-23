import cv2 
import numpy as np 
import os 
'''
show the truth class and bbox annotation ,check the converted cocoformat annotation 
'''
import cv2 
import numpy as np 
import os 
'''
show the truth class and bbox annotation ,check the converted cocoformat annotation 
'''

def get_image(img_meta,coco):
 img = coco.loadImgs(img_meta)
 #print img
 img_name = img[0]['file_name']
 return img_name

def draw_bbox_and_class(coco,img_name,image_id,save_dir,thick=1):
 image = cv2.imread(img_dir+img_name)
 box_anno_id = coco.getAnnIds(imgIds=image_id)
 truth_box_anno = coco.loadAnns(box_anno_id)
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
 all_image_id = coco.getImgIds()
 for j in all_image_id:
  print('process the img = {}'.format(j))
  img_name = get_image(j,coco)
  draw_bbox_and_class(coco,img_name,j,save_dir)

if __name__ == '__main__':
 anno_dir = '/data1/shuai/adas/annotations/img_day_train_22000.json'
 img_dir = '/data1/shuai/adas/img_day/'
 save_dir = '/data1/shuai/adas/test/'
 main()
def get_image(img_meta,coco):
 img = coco.loadImgs(img_meta)
 #print img
 img_name = img[0]['file_name']
 return img_name

def draw_bbox_and_class(coco,img_name,image_id,save_dir,thick=1):
 image = cv2.imread(img_dir+img_name)
 box_anno_id = coco.getAnnIds(imgIds=image_id)
 truth_box_anno = coco.loadAnns(box_anno_id)
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
 all_image_id = coco.getImgIds()
 for j in all_image_id:
  print('process the img = {}'.format(j))
  img_name = get_image(j,coco)
  draw_bbox_and_class(coco,img_name,j,save_dir)

if __name__ == '__main__':
 anno_dir = '/data1/shuai/adas/annotations/img_day_train_22000.json'
 img_dir = '/data1/shuai/adas/img_day/'
 save_dir = '/data1/shuai/adas/test/'
 main()
