import numpy as np 
import cv2 
import glob 
import argparse
import os 
import sys 
from caffe2.python import workspace
core_root = '/home/wangshuai/seg_every_thing-master/lib'
sys.path.insert(0,core_root)
from caffe2.python import core

from core.config import assert_and_infer_cfg
from core.config import cfg 
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
from core.test_engine import initialize_model_from_cfg
import utils.c2 as c2_utils
import utils.blob as blob_utils
import utils.image as image_utils
import utils.net as net_utils
from modeling import model_builder

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)

def parse_args():
 parser = argparse.ArgumentParser(description='see the bolb in mask branch')
 parser.add_argument('--cfg',dest='cfg',type=str)
 parser.add_argument('--wts',dest='wts',type=str)
 parser.add_argument('--output-dir',dest='output-dir',type=str)
 parser.add_argument('--img',dest='img',type=str)
 if len(sys.argv) == 1:
  parser.print_help()
  sys.exit(1)
 return parser.parse_args()

def get_model(cfg_file, weights_file):  
    merge_cfg_from_file(cfg_file)  
    cfg.TRAIN.WEIGHTS = ''  # NOTE: do not download pretrained model weights  
    cfg.TEST.WEIGHTS = weights_file  
    cfg.NUM_GPUS = 1  
    assert_and_infer_cfg() 
    #according the cfg to bulid model
    model = initialize_model_from_cfg(weights_file)  
    return model  

if __name__ == '__main__':
 workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
 args = parse_args()
 model = get_model(args.cfg, args.wts)
 img = cv2.imread(args.img)
 #im_scale = im_conv_body_only(model,img,cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
 im_blob, im_scale, _im_info = blob_utils.get_image_blob(
  img, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
 with c2_utils.NamedCudaScope(0):
 #  workspace.FeedBlob(core.ScopedName('data'), im_blob)
 #  workspace.RunNet(model.net.Proto().name)
 #  blob = workplace.FetchBlob('rois')
 # print 1
  cls_b,_,_ = infer_engine.im_detect_all(model,img,None)
  blobs = workspace.Blobs()
  print blobs
  mask_logits = workspace.FetchBlob(core.ScopedName('mask_logits'))
  #print mask_logits
  print mask_logits.shape
 np.save('/data1/shuai/adas/code/mask_logits.npy',mask_logits)
