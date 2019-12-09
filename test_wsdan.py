"""EVALUATION
Created: Nov 22,2019 - Yuchong Gu
Revised: Nov 29,2019 - Yuchong Gu
"""
import os
import logging
import warnings
from optparse import OptionParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np


from models import WSDAN
from dataset import *
from utils import *
from visuals.generate_confusion_matrix import draw_confusion_matrix
# visualize
visualize = False
savepath = 'temp/'
if visualize:
    os.makedirs(savepath, exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)



def main():
    parser = OptionParser()
    
    parser.add_option('--gpu', '--gpu', dest='GPU', default=0, type='int',
                      help='GPU Id (default: 0)')
    parser.add_option('--evalckpt', '--eval-ckpt', dest='eval_ckpt', default='models/wsdan/003.ckpt',
                      help='saved models are in ckpt directory')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=64, type='int',
                      help='batch size (default: 16)')
    parser.add_option('-j', '--workers', dest='workers', default=4, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('--na', '--num-attentions', dest='num_attentions', default=32, type='int',
                      help='number of attentions')
    parser.add_option('--cm', '--confusion_matrix', dest='confusion_matrix', default=True,
                      help='if you want to create confusion matrix')
    
    (options, args) = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")
    try:
        ckpt = options.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation options')
        return
    # Dataset for testing
    transform = transforms.Compose([transforms.Resize(size=(256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    
    test_dataset = CustomDataset(data_root='/mnt/HDD/DatasetOriginals/RFW/test/data/',csv_file='data/RFW_Test_Images_Metadata.csv',transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=options.batch_size * 4, shuffle=False,num_workers=options.workers, pin_memory=True)


    ##################################
    # Initialize model
    ##################################
    net = WSDAN(num_classes=4, M=32, net='inception_mixed_6e')

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))
    ##################################
    # use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Prediction
    ##################################
    raw_accuracy = TopKAccuracyMetric(topk=(1, 3))
    ref_accuracy = TopKAccuracyMetric(topk=(1, 3))
    raw_accuracy.reset()
    ref_accuracy.reset()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top_refined = AverageMeter('Acc@1', ':6.2f')
    net.eval()
    y_pred,y_true = [],[]
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, (X, y) in enumerate(test_loader):
            y_true += list(y.numpy())
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

            y_pred_crop, _, _ = net(crop_image)
            y_predicted = (y_pred_raw + y_pred_crop) / 2.
            _, pred = y_predicted.topk(1, 1, True, True)

            y_pred += list(pred.cpu().numpy())

            if visualize:
                # reshape attention maps
                attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * options.batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * options.batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * options.batch_size + batch_idx)))

            # Top K
            epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            epoch_ref_acc = ref_accuracy(y_predicted, y)
            top1.update(epoch_raw_acc[0], X.size(0))
            top_refined.update(epoch_ref_acc[0], X.size(0))

            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}, {:.2f}), Refine ({:.2f}, {:.2f})'.format(
                epoch_raw_acc[0], epoch_raw_acc[1], epoch_ref_acc[0], epoch_ref_acc[1])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        pbar.close()
        print(' * Raw Accuracy {top1.avg:.3f}'.format(top1=top1))
        print(' * Refined Accuracy {top1.avg:.3f}'.format(top1=top_refined))
        print(len(y_pred),len(y_true))
        if options.confusion_matrix:
            file_name ='source/wsdan_confusion_matrix.svg'
            draw_confusion_matrix(np.asarray(y_true),np.asarray(y_pred),file_name)


if __name__ == '__main__':
    main()