import pickle

import numpy as np

from eval import eval_p
from models import model
from video_dataset_anomaly_balance_uni_sample import dataset
from torch.utils.data import DataLoader
import time
import torch
import os
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='MSTR_Net')
# settings for dataset
parser.add_argument('--dataset_name', type=str, default='ucfcrime', help='')
parser.add_argument('--dataset_path', type=str, default='',
                        help='path to dir contains anomaly datasets')
parser.add_argument('--ckpt_path', type=str, default='',
                        help='path to best ckpt')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')

parser.add_argument('--feature_modal', type=str, default='rgb',
                        help='features from different input, options contain rgb, flow , combine')
# settings for model MSTR
parser.add_argument('--Vitblock_num', type=int, default=6, help='1-8')
parser.add_argument('--cross_clip', type=int, default=4, help='1,2,4')
parser.add_argument('--plot', type=int, default=1, help='0,1')

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join('result',args.dataset_name)
    #load ckpt
    test_dataset = dataset(args=args, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    model = model(0, feature_size=2048, Vitblock_num=args.Vitblock_num,
                  cross_clip=1, split=0, beta=0, delta=0)
    trained_weight = torch.load(args.ckpt_path)
    model.load_state_dict(trained_weight)
    model.to(device)
    with open(file='dataset/'+args.dataset_name+'/GT/frame_label.pickle', mode='rb') as f:
        label_dict = pickle.load(f)
        if type(label_dict) is list:
            label_dict = np.array(label_dict)
    #inference
    model.eval()
    with torch.no_grad():
        total_time = time.time()
        result = {}
        for i, data in enumerate(test_loader):
            feature, data_video_name = data
            feature = feature.squeeze(2).to(device)
            with torch.no_grad():
                # if data_video_name[0] == '01_0135':
                #     num = model(feature, False, is_training=True,is_test=True).cpu().detach().numpy()
                #     np.save('atten_map/mstr01_0135.npy',num)
                element_logits = model(feature, None, None, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)
            result[data_video_name[0]] = element_logits
        total_time = time.time()-total_time
        eval_p(itr='inference', dataset=args.dataset_name, predict_dict=result, logger=None, save_path=save_path,
               plot=True, args=args)

        print('{} inf_time {}'.format(args.dataset_name,total_time))

