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
from utils import accuracy, batch_augment

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# visualize
visualize = False
savepath = 'visulize/'
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
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
    parser.add_option('--evalckpt', '--eval-ckpt', dest='eval_ckpt', default='models/008.ckpt',
                      help='saved models are in ckpt directory')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                      help='batch size (default: 16)')
    parser.add_option('-j', '--workers', dest='workers', default=4, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('--na', '--num-attentions', dest='num_attentions', default=32, type='int',
                      help='number of attentions')
    parser.add_option('--net', '--net', dest='net', default='inception_mixed_6e',
                      help='it can be wsdan,resnet50,resnet100,inception')
    

    (options, args) = parser.parse_args()

    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")


    try:
        ckpt = options.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation options')
        return

    ##################################
    # Dataset for testing
    ##################################
    transform = transforms.Compose([transforms.Resize(size=(400,400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])])
    
    test_dataset = CustomDataset(data_root='/mnt/HDD/DatasetOriginals/RFW/test/data/',csv_file='data/RFW_Test_Images_Metadata_cleaner.csv',transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=options.batch_size * 4, shuffle=False,num_workers=options.workers, pin_memory=True)


    ##################################
    # Initialize model
    ##################################
    net = WSDAN(num_classes=4, M=options.num_attentions)

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    cudnn.benchmark = True
    net.to(device)
    net = nn.DataParallel(net)
    net.eval()

    ##################################
    # Prediction
  

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        avg_accuracy=0
        ref_accuracy=0
        results = []
        ground_truths = []
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            y_pred_raw, _, attention_maps = net(X)

            # Augmentation with crop_mask
            crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1)

            y_pred_crop, _, _ = net(crop_image)
            y_pred = (y_pred_raw + y_pred_crop) / 2.

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
            epoch_raw_acc,pred_list= accuracy(y_pred_raw, y,topk=(1, 3))
            epoch_ref_acc,_ = accuracy(y_pred, y,topk=(1, 3))
            results += list(pred_list[0].cpu().numpy())
            ground_truths += list(y.cpu().numpy())
            avg_accuracy += epoch_raw_acc[0].item() 
            ref_accuracy += epoch_ref_acc[0].item() 
            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}, {:.2f}), Refine ({:.2f}, {:.2f})'.format(
                epoch_raw_acc[0], epoch_raw_acc[1], epoch_ref_acc[0], epoch_ref_acc[1])
            pbar.update()
            pbar.set_postfix_str(batch_info)

        print('avg raw accuracy',avg_accuracy/(i+1))
        print('avg ref accuracy',ref_accuracy/(i+1))
        print(len(results))
        np.save('wsdan_ground_truth.npy',np.asarray(ground_truths))
        np.save('wsdan_predictions.npy',np.asarray(results))

        pbar.close()


if __name__ == '__main__':
    main()