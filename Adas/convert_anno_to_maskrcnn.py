import json
import numpy as np
from pycocotools.coco import COCO  
import pycocotools.mask as mask_util
import cv2

"""
convert the maskxrcnn output annotation to maskrcnn annotation
find the iou>0.7(gtbox and dtbox),displace the dtbox with gtbox
combine the box and the mask to cocoformat 
"""
def tran_dtbox_to_gt_box(gt_box_file,dt_box_file):
 """
 displace the dtbox with gtbox
 gt_box_file: the gt coco format annotation
 dt_box_file: the bbox output of maskrcnn
 """
 gt_annotations = COCO(gt_box_file)
 dt_boxes_annos = json.load(open(dt_box_file))
 result = []
 
 for j,dt_box_anno in enumerate(dt_boxes_annos):
  #dt threshold
  if dt_box_anno['score'] < 0.9:
   result.append('del')
   continue
  ious = []
  img_id = dt_box_anno['image_id']
  category_id = dt_box_anno['category_id']
  gt_boxes_ids = gt_annotations.getAnnIds(imgIds=img_id,catIds=category_id)
  gt_boxes_anno =  gt_annotations.loadAnns(gt_boxes_ids)
  for i in gt_boxes_anno:
   gt_box = i['bbox']
   dt_box = dt_box_anno['bbox']
   iou = iou_compute(gt_box,dt_box)
   ious.append(iou)
  if len(ious) == 0:
   result.append('del')
  #iou
  elif max(ious) > 0.7:
   id = np.argsort(np.asarray(ious))[-1]
   result.append(gt_boxes_anno[id])
  else:
   result.append('del')
 return result

def blend_box_mask(box_anno,mask_anno_file,raw_file,result_file):
 """
 displace the segmentation[[1,1,1,1,1,1]] to the maskrcnn output mask
 convert the rle mask to poly mask for train
 form the cocoformat for train
 """

 raw_file = json.load(open(raw_file))
 mask_anno = json.load(open(mask_anno_file))
 annotations = raw_file['annotations']
 categories = raw_file['categories']
 images = raw_file['images']
 annotations = []
 for i,box in enumerate(box_anno):
  print i 
  if box == 'del':
   continue
  else:
   #trans rle mask to poly
   mask = mask_anno[i]['segmentation']
   mask = mask_util.decode(mask)
   mask_ = mask.astype(np.uint8).copy() 
   mask_new,contours,hierarchy = cv2.findContours(mask_,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   segmentation = []
   for contour in contours:
    contour = contour.flatten().tolist()
    if len(contour) > 4:
     segmentation.append(contour)
   #
   box['segmentation'] = segmentation
   annotations.append(box)
 final = {'images':images,'annotations':annotations,'categories':categories}
 print('number of dt_box {}'.format(len(box_anno)))
 print('number of dt_mask {}'.format(len(mask_anno)))
 print('len of box and mask {}'.format(len(annotations)))
 
 with open(result_file,'w') as f:
  json.dump(final,f)


def iou_compute(gt_box,dt_box):
 """
 input: string list of cordination 
 return: list of iou
 """
 gt_box_1_x = int(gt_box[0])
 gt_box_1_y = int(gt_box[1])
 gt_box_2_x = int(gt_box[0]) + int(gt_box[2])
 gt_box_2_y = int(gt_box[1]) + int(gt_box[3])
 dt_box_1_x = int(dt_box[0])
 dt_box_1_y = int(dt_box[1])
 dt_box_2_x = int(dt_box[0]) + int(dt_box[2])
 dt_box_2_y = int(dt_box[1]) + int(dt_box[3])
 #intersect
 inter_1_x = max(gt_box_1_x,dt_box_1_x)
 inter_1_y = max(gt_box_1_y,dt_box_1_y)
 inter_2_x = min(gt_box_2_x,dt_box_2_x)
 inter_2_y = min(gt_box_2_y,dt_box_2_y)
 #print(inter_1_x,inter_1_y)
 #print(inter_2_x,inter_2_y)
 if (inter_2_x >= inter_1_x) and (inter_2_y >= inter_1_y):
  intersect = (inter_2_y - inter_1_y)*(inter_2_x - inter_1_x)
 else:
  intersect = 0
 #iou
 #print('intersect',intersect)
 iou = (float(intersect) / (int(gt_box[2])*int(gt_box[3])+int(dt_box[2])*int(dt_box[3])-intersect))
 return iou 
if __name__ == '__main__':
 mask_anno_file = '/data1/shuai/adas/detectron_output/test/adas_train_20171026/mask_rcnn/segmentations_adas_train_20171026_results.json' 
 raw_file = '/data1/shuai/adas/annotations/20171026.json'
 result_file = '/data1/shuai/adas/annotations/20171026_maskx_to_mask.json'
 gt_box_file = '/data1/shuai/adas/annotations/20171026.json'
 dt_box_file = '/data1/shuai/adas/detectron_output/test/adas_train_20171026/mask_rcnn/bbox_adas_train_20171026_results.json'
 box_anno = tran_dtbox_to_gt_box(gt_box_file,dt_box_file)
 blend_box_mask(box_anno,mask_anno_file,raw_file,result_file)
