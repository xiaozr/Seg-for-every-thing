"""
adas's annotation is txt which has bbox information
need to transfer coco_format for train
we only need the first 5 classes
"""

import os 
import json 

def main():
"""
roidb_file: rodb.txt(box_nums x1,y1,x2,y2,class,x1,y1,x2,y2,class,...
                     box_nums x1,y1,x2,y2,class,x1,y1,x2,y2,class,...) 
trainval_file: trainval.txt(img_name
                            img_name)
  
"""

	trainval_file = open(data_root+'trainval.txt')
	roidb_file = open(data_root+'roidb.txt')
	images = []
	annotations = []
	roi_id = 0
	img_id = 0
	for roidbs in roidb_file:
		#list of 'images'
		file_name = trainval_file.readline().split('\r\n')[0] + '.jpg'
		print file_name
		width = 1280
		height = 720
		image = {"file_name":file_name,'width':width,'height':height,'id':img_id}
		images.append(image)
		#list of 'annotations'
		size = [height,width]
		iscrowd = 0
		roidbs = roidbs.split(' ')
		for j in range(int(roidbs[0])):
			category_id = int(roidbs[5*j+5])
			#the first 5 class
                        if category_id > 5:
				continue
			idx = int(roidbs[5*j+1])
			idy = int(roidbs[5*j+2])
			idw = int(roidbs[5*j+3]) - int(roidbs[5*j+1])
			idh = int(roidbs[5*j+4]) - int(roidbs[5*j+2])
			bbox = [idx,idy,idw,idh]
			annotation = {'size':size,'iscrowd':0,'image_id':img_id,'bbox':bbox,'category_id':category_id,'id':roi_id}
			annotations.append(annotation)
			roi_id = roi_id + 1
		img_id = img_id + 1
		#list of 'categories'
		categories = [{'id':1,'name':'car'},{'id':2,'name':'bus'},{'id':3,'name':'truck'},{'id':4,'name':'person'},{'id':5,'name':'non_motor_vehicle'}]
		
	#final json 
	final = {'images':images,'annotations':annotations,'categories':categories}
	with open(result_dir + result_json_name,'w') as f:
		json.dump(final,f)


if __name__ == '__main__':
	data_root = '/data1/shuai/adas/20171023/'
	result_dir = '/data1/shuai/adas/annotations/'
	result_json_name = '20171023.json'
	main()
