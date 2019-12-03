import os
import pandas as pd
import numpy as np
import torch
import logging
import time
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)

##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - 0.05 * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + 0.05 * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - 0.05 * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + 0.05 * imgW), imgW)

            crop_images.append(
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size))

    return np.array(res)



def test_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size))

    return np.array(res),pred
def generate_data_svg_plots(data_mode,metadata_file):
    data = pd.read_csv(metadata_file)
    data =data.sort_values(by=['race'])
    
    fig = px.histogram(data, x="race", 
                    histfunc="sum", 
                    color=data.race,
                    color_discrete_sequence=['#ED553B','#F6D55C','#3CAEA3','#20639B'], 
                    barmode="group")

    fig.update_yaxes(title_text="Number of Images")
    fig.update_xaxes(title_text="Race")
    fig.update_layout(title_text="RFW %s Image's Race Distribution"%data_mode)
    file_name = "source/RFW_{}_Images_Race_Distribution.svg".format(data_mode)
    fig.write_image(file_name)

# generate_data_svg_plots('Train','data/RFW_Train40k_Images_Metada.csv')
# generate_data_svg_plots('Test','data/RFW_Test_Images_Metadata.csv')
# generate_data_svg_plots('Validation','data/RFW_Val4k_Images_Metadata.csv')