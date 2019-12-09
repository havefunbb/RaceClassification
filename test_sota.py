from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
import warnings
from train_sota import initialize_model
from utils import *
from visuals.generate_confusion_matrix import draw_confusion_matrix

from optparse import OptionParser
from dataset import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = OptionParser()
    parser.add_option('--model', '--model-name', dest='model_name', default='wsdan',
                      help='it can be wsdan,resnet50,resnet101,inception')
    parser.add_option('--ckpt', '--ckpt_file', dest='ckpt_file', default='models/wsdan/001.ckpt',
                      help='check point file')
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                      help='batch size (default: 16)')
    
    parser.add_option('--cm', '--confusion_matrix', dest='confusion_matrix', default=True,
                      help='if you want to create confusion matrix')
    
    (options, args) = parser.parse_args()
    
    #step 1 model load
    
    num_classes=4
    model,input_size = initialize_model(options.model_name,num_classes,feature_extract=False, use_pretrained=False)
    criteration = nn.CrossEntropyLoss()
    model = load_checkpoint(options.ckpt_file,model)
    model = model.to(device)
    #sstep 2 dataset initilization
    transform = transforms.Compose([transforms.Resize(size=input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    test_dataset = CustomDataset(data_root='/mnt/HDD/RFW/test/data/',csv_file='data/RFW_Test_Images_Metadata.csv',transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size * 4, shuffle=False,num_workers=options.workers, pin_memory=True)
    _,y_true,y_pred = test(test_loader,model,criteration)
    if options.confusion_matrix:
        file_name ='source/%s_confusion_matrix.svg'%(options.model_name)
        draw_confusion_matrix(y_true,y_pred,file_name)
    #step 3 test model with test set
    
def load_checkpoint(ckpt_file,model):
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}'".format(ckpt_file))
        if device is None:
            checkpoint = torch.load(ckpt_file)
        else:
                # Map model to be loaded to specified single gpu.
            loc = device
            checkpoint = torch.load(ckpt_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_file))
    return model

def test(test_loader, model, criterion):
    y_pred,y_true = [],[]
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top3 = AverageMeter('Acc@3', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top3],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            y_true += list(target.numpy())
            if device is not None:
                images = images.cuda(device, non_blocking=True)
                target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc3 = accuracy(output, target, topk=(1, 3))
            _, pred = output.topk(1, 1, True, True)
            y_pred += list(pred.cpu().numpy())
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top3.update(acc3[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))
        print(len(y_pred),len(y_true))

    return top1.avg,np.asarray(y_true),np.asarray(y_pred)



if __name__ == '__main__':
    main()