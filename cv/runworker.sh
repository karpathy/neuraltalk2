#!/bin/bash

echo "worker $1 is starting. Exporting LD_LIBRARY_PATH then running driver.py"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCHPATH/lib:/usr/local/lib:/usr/local/cuda/lib64:/home/stanford/cudnn_r3:/home/stanford/cudnn_r3/lib64
python driver.py $1
