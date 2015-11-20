"""
This script should be run from root directory of this codebase:
https://github.com/tylin/coco-caption
"""

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys

input_json = sys.argv[1]


annFile = 'annotations/captions_val2014.json'
coco = COCO(annFile)
valids = coco.getImgIds()

checkpoint = json.load(open(input_json, 'r'))
preds = checkpoint['val_predictions']

# filter results to only those in MSCOCO validation set (will be about a third)
preds_filt = [p for p in preds if p['image_id'] in valids]
print 'using %d/%d predictions' % (len(preds_filt), len(preds))
json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

resFile = 'tmp.json'
cocoRes = coco.loadRes(resFile)
cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.params['image_id'] = cocoRes.getImgIds()
cocoEval.evaluate()

# create output dictionary
out = {}
for metric, score in cocoEval.eval.items():
    out[metric] = score
# serialize to file, to be read from Lua
json.dump(out, open(input_json + '_out.json', 'w'))

