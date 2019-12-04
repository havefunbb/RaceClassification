#!/bin/sh

python train_sota.py --model resnet50 2>&1 | tee resnet50.txt
python train_sota.py --model resnet100 2>&1 | tee resnet100.txt
python train_sota.py --model inception 2>&1 | tee inception.txt