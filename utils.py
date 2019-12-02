import os
import pandas as pd
import numpy as np
import torch
import logging
import time
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px

logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)




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


def generate_data_svg_plots(data_mode,metadata_file):
    data = pd.read_csv(metadata_file)
    
    fig = px.histogram(data, x="race", 
                    histfunc="sum", 
                    color="race",
                    color_discrete_sequence=['#ED553B','#F6D55C','#3CAEA3','#20639B'], 
                    barmode="group")

    fig.update_yaxes(title_text="Number of Images")
    fig.update_xaxes(title_text="Race")
    fig.update_layout(title_text="RFW %s Image's Race Distribution"%data_mode)
    file_name = "source/RFW_{}_Images_Race_Distribution.svg".format(data_mode)
    fig.write_image(file_name)

generate_data_svg_plots('Train','data/RFW_Train40k_Images_Metada.csv')
generate_data_svg_plots('Test','data/RFW_Test_Images_Metadata.csv')
generate_data_svg_plots('Validation','data/RFW_Val4k_Images_Metadata.csv')