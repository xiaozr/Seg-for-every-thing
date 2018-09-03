from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
from __future__ import unicode_literals  
 
import argparse  
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)  
import sys  
 
from caffe2.python import net_drawer  

#add the path of core
core_root = '/home/wangshuai/seg_every_thing-master/lib'
sys.path.insert(0,core_root)

from core.config import assert_and_infer_cfg  
from core.config import cfg  
from core.config import merge_cfg_from_file  
import utils.train as train_utils  
import utils.c2 as c2_utils  
from modeling import model_builder
  
c2_utils.import_detectron_ops()  

# OpenCL may be enabled by default in OpenCV3; disable it because it's not  
# thread safe and causes unwanted GPU memory allocations.  
cv2.ocl.setUseOpenCL(False)  

def parse_args():  
    parser = argparse.ArgumentParser(description='Network Visualization')  
    parser.add_argument(  
        '--cfg',  
        dest='cfg',  
        help='cfg model file (/path/to/model_config.yaml)',  
        default=None,  
       type=str  
    )  
    parser.add_argument(  
        '--wts',  
        dest='weights',  
        help='weights model file (/path/to/model_weights.pkl)',  
        default=None,  
        type=str  
    )  
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
    model = model_builder.create(cfg.MODEL.TYPE,train=True)  
    return model  
  
  
if __name__ == '__main__':  
    args = parse_args()  
    #get the model 
    model = get_model(args.cfg, args.weights)
    print(model.net)
    ops = model.net.Proto().op
    #ops is protobuf object
    print(len(ops),type(ops))
    for i in range(len(ops)):
        output_name = str(ops[i].output[0]) 
        #remove the all op about grad
        if str(ops[i].type) == 'SigmoidCrossEntropyLossGradient':
            break
    ops = ops[:i]
    g = net_drawer.GetPydotGraph(ops, rankdir="TB")
    #g = net_drawer.GetPydotGraphMinimal(ops, rankdir="TB")  
    g.write_png(model.Proto().name + 'test.png')  
