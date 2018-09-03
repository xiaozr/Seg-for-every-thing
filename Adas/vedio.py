# encoding: UTF-8
import cv2
import os
from PIL import Image ,ImageDraw,ImageFont
import shutil
img_path1 = "/tmp/detectron-visualizations"

out_folder = "/data1/shuai/adas/code/video"
if os.path.exists(out_folder):
 shutil.rmtree(out_folder)
os.makedirs(out_folder)
videoWriter = cv2.VideoWriter('/data1/shuai/adas/code/video/result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280,720))

EXTENSIONS = ['.jpg','.png']
def is_image(filename):
 return any(filename.endswith(ext) for ext in EXTENSIONS)

filenames = [f for f in sorted(os.listdir(img_path1)) if is_image(f)]
filenames = sorted(filenames,key= lambda x:int(x.split('_')[-1].split('.')[0]))[1:3500]

# filenames2 = [f for f in sorted(os.listdir(img_path2)) if is_image(f)]
# filenames2 = sorted(filenames2,key= lambda x:int(x.split('_')[-1].split('.')[0]))

# filenames3 = [f for f in sorted(os.listdir(img_path3)) if is_image(f)]
# filenames3 = sorted(filenames,key= lambda x:int(x.split('_')[-1].split('.')[0]))

# filenames4 = [f for f in sorted(os.listdir(img_path4)) if is_image(f)]
# filenames4 = sorted(filenames2,key= lambda x:int(x.split('_')[-1].split('.')[0]))

for i in range(len(filenames)): 
    # concat picture
   # print filenames[i].split('_')[-1].split('.')[0]
    
    #ttfont = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 24)
    print i 
    img1  = cv2.imread(os.path.join(img_path1,filenames[i])) 
    #draw = ImageDraw.Draw(img1)
    #draw.text((10,10),u'segnet_360_360', fill=(0,0,0),font=ttfont)
    # img2  = Image.open(os.path.join(img_path2,filenames2[i])) 
    # draw = ImageDraw.Draw(img2)
    # draw.text((10,10),u'res_seg_360_360', fill=(0,0,0),font=ttfont)
    # img3  = Image.open(os.path.join(img_path3,filenames[i])) 
    # draw = ImageDraw.Draw(img3)
    # draw.text((10,10),u'segnet_720_720', fill=(0,0,0),font=ttfont)
    # img4  = Image.open(os.path.join(img_path4,filenames2[i])) 
    # draw = ImageDraw.Draw(img4)
    # draw.text((10,10),u'res_seg_720_720', fill=(0,0,0),font=ttfont)
 
    #result = Image.new(img1.mode,(width*2,height*2))
    #result.paste(img1)
    # result.paste(img2,(width,0,2*width,height))
    # result.paste(img3,(0,height,width,2*height))
    # result.paste(img4,(width,height,2*width,2*height))
    
    #result.save(os.path.join(out_folder, filenames[i]))
    #img = cv2.imread(os.path.join(out_folder,filenames[i]))
    #print img.shape
    videoWriter.write(img1)
