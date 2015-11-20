import os
from random import uniform, randrange, choice
import math
import time
import sys
import json

def encodev(v):
  if isinstance(v, float):
    return '%.3g' % v
  else:
    return str(v)

assert len(sys.argv) > 1, 'specify gpu/rnn_size/num_layers!'
gpuid = int(sys.argv[1])

cmd = 'CUDA_VISIBLE_DEVICES=%d th train.lua ' % (gpuid, )
while True:
  time.sleep(1.1+uniform(0,1))

  opt = {}
  opt['id'] = '%d-%0d-%d' % (gpuid, randrange(1000), int(time.time()))
  opt['gpuid'] = 0
  opt['seed'] = 123
  opt['val_images_use'] = 3200
  opt['save_checkpoint_every'] = 2500

  opt['max_iters'] = -1 # run forever
  opt['batch_size'] = 16

  #opt['checkpoint_path'] = 'checkpoints'

  opt['language_eval'] = 1 # do eval

  opt['optim'] = 'adam'
  opt['optim_alpha'] = 0.8
  opt['optim_beta'] = choice([0.995, 0.999])
  opt['optim_epsilon'] = 1e-8
  opt['learning_rate'] = 10**uniform(-5.5,-4.5)

  opt['finetune_cnn_after'] = -1 # dont finetune
  opt['cnn_optim'] = 'adam'
  opt['cnn_optim_alpha'] = 0.8
  opt['cnn_optim_beta'] = 0.995
  opt['cnn_learning_rate'] = 10**uniform(-5.5,-4.25) 

  opt['drop_prob_lm'] = 0.5

  opt['rnn_size'] = 512
  opt['input_encoding_size'] = 512

  opt['learning_rate_decay_start'] = -1 # dont decay
  opt['learning_rate_decay_every'] = 50000

  opt['input_json'] = '/scr/r6/karpathy/cocotalk.json'
  opt['input_h5'] = '/scr/r6/karpathy/cocotalk.h5'

  #opt['start_from'] = '/scr/r6/karpathy/neuraltalk2_checkpoints/good6/model_id0-565-1447975213.t7'

  optscmd = ''.join([' -' + k + ' ' + encodev(v) for k,v in opt.iteritems()])
  exe = cmd + optscmd + ' | tee /scr/r6/karpathy/neuraltalk2_checkpoints/out' + opt['id'] + '.txt'
  print exe
  os.system(exe)

