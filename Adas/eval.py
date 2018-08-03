"""
eval every category of the test data 
"""
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt=COCO('/data1/shuai/adas/annotations/20171023_500.json')
cocoDt=cocoGt.loadRes('/data1/shuai/adas/detectron_output/test/adas_train_20171023_500/mask_rcnn/bbox_adas_train_20171023_500_results.json')
cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
for i in range(5):
 cocoEval.params.catIds = [i+1]
 cocoEval.evaluate()
 cocoEval.accumulate()
 cocoEval.summarize()
