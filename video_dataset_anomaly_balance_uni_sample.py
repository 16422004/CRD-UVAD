import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import utils
import options
import os
import pickle
import random
#import cupy

class train_loader():
    def __init__(self, args, train=True, trainlist=None, testlist=None):
        args.dataset_path = './dataset'
        self.args = args
        self.cross_clip = args.cross_clip
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        # tcc
        self.trainlist = self.txt2list(
            txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_split.txt'))
        self.video_train = self.trainlist
        self.train = train
        self.t_max = args.max_seqlen
        self.pretrain_result_dict = None
        self.max_seqlen = args.max_seqlen
        self.train_features = self.feature_all(feature_path=self.feature_path, trainlist=self.trainlist,cross_clip=self.cross_clip)
        self.size = self.train_features.shape[0]//(self.max_seqlen//self.cross_clip)

    def feature_all(self, feature_path, trainlist, cross_clip):
        all_features = None
        for i,data in enumerate(trainlist):
            infile_path = os.path.join(feature_path, str(data.replace('\n', '')), 'feature.npy')
            feature = np.load(infile_path)
            #for ucfcrime:
            if len(feature.shape) == 3 and feature.shape[1] == 1:
                feature = np.squeeze(feature,1)
            elif len(feature.shape) == 3 and feature.shape[0] == 1:
                feature = np.squeeze(feature,0)
            feature_size = feature.shape[0]
            feature1 = np.expand_dims(feature[0:feature_size-cross_clip+1], axis=1)
            for k in range(1,cross_clip):
                feature2 = np.expand_dims(feature[k:feature_size-cross_clip+1+k], axis=1)
                feature1 = np.concatenate((feature1,feature2),axis=1)
            # if self.dataset_name == 'ucfcrime':
            #     if feature1.shape[0] > 100:
            #         feature = None
            #         if feature1.shape[0] > 10000:
            #             for j in range(feature1.shape[0]//100):
            #                 if feature is None:
            #                     feature = np.expand_dims(feature1[j * 100],axis=0)
            #                 else:
            #                     feature = np.append(feature, np.expand_dims(feature1[j * 100],axis=0),axis=0)
            #         else:
            #             for j in range(feature1.shape[0] // 10):
            #                 if feature is None:
            #                     feature = np.expand_dims(feature1[j * 10],axis=0)
            #                 else:
            #                     feature = np.append(feature, np.expand_dims(feature1[j * 10],axis=0), axis=0)
            #     else:
            #         feature = feature1
            # else:
            #     feature = feature1
            feature = feature1
            if i == 0:
                all_features = feature
            else:
                all_features = np.append(all_features, feature, axis=0)
        return all_features

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def getitem(self,itr):
        data = self.train_features[itr:itr+(self.max_seqlen//self.cross_clip)]
        return data

class dataset(Dataset):
    def __init__(self, args, train=False, trainlist=None, testlist=None):
        """
        :param args:
        self.dataset_path: path to dir contains anomaly datasets
        self.dataset_name: name of dataset which use now
        self.feature_modal: features from different input, contain rgb, flow or combine of above type
        self.feature_pretrain_model: the model name of feature extraction
        self.feature_path: the dir contain all features, use for training and testing
        self.videoname: videonames of dataset
        self.trainlist: videonames of dataset for training
        self.testlist: videonames of dataset for testing
        self.train: boolen type, if it is True, the dataset class return training data
        self.t_max: the max of sampling in training
        """
        args.dataset_path = './dataset'
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             args.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        #tcc
        self.testlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'test_split.txt'))
        self.video_label_dict = self.pickle_reader(
            file=os.path.join(self.dataset_path, self.dataset_name, 'GT', 'video_label.pickle'))
        self.train = train
        self.pretrain_result_dict = None
        self.video_labels = None

    def data_dict_creater(self):
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train

    def __getitem__(self, index):

        data_video_name = self.testlist[index].replace('\n', '').replace('Ped', 'ped')
        self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
        if len(self.feature.shape) == 3 and self.feature.shape[1] == 1:
            self.feature = np.squeeze(self.feature, 1)
        elif len(self.feature.shape) == 3 and self.feature.shape[0] == 1:
            self.feature = np.squeeze(self.feature, 0)
        return self.feature, data_video_name

    def __len__(self):
        if self.train:
            return len(self.trainlist)

        else:
            return len(self.testlist)


class dataset_train2test(Dataset):
    def __init__(self, args, trainlist=None):
        self.args = args
        self.dataset_path = args.dataset_path
        self.dataset_name = args.dataset_name
        self.feature_modal = args.feature_modal
        self.feature_pretrain_model = args.feature_pretrain_model
        if self.feature_pretrain_model == 'c3d' or self.feature_pretrain_model == 'c3d_ucf':
            self.feature_layer = args.feature_layer
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_layer, self.feature_modal)
        else:
            self.feature_path = os.path.join(self.dataset_path, self.dataset_name, 'features_video',
                                             self.feature_pretrain_model, self.feature_modal)
        self.videoname = os.listdir(self.feature_path)
        if self.args.larger_mem:
            self.data_dict = self.data_dict_creater()
        if trainlist:
            self.trainlist = self.txt2list(trainlist)
        else:
            self.trainlist = self.txt2list(txtpath=os.path.join(self.dataset_path, self.dataset_name, 'train_occ.txt'))
    def data_dict_creater(self):
        data_dict = {}
        for _i in self.videoname:
            data_dict[_i] = np.load(
                file=os.path.join(self.feature_path, _i.replace('\n', '').replace('Ped', 'ped'), 'feature.npy'))
        return data_dict

    def txt2list(self, txtpath=''):
        """
        use for generating list from text file
        :param txtpath: path of text file
        :return: list of text file
        """
        with open(file=txtpath, mode='r') as f:
            filelist = f.readlines()
        return filelist

    def pickle_reader(self, file=''):
        with open(file=file, mode='rb') as f:
            video_label_dict = pickle.load(f)
        return video_label_dict

    def p_n_split_dataset(self, video_label_dict, trainlist):
        normal_video_train = []
        anomaly_video_train = []
        for t in trainlist:
            if video_label_dict[t.replace('\n', '').replace('Ped', 'ped')] == [1.0]:
                anomaly_video_train.append(t.replace('\n', ''))
            else:
                normal_video_train.append(t.replace('\n', '').replace('Ped', 'ped'))
        return normal_video_train, anomaly_video_train


    def __getitem__(self, index):
            data_video_name = self.trainlist[index].replace('\n', '').replace('Ped', 'ped')
            self.feature = np.load(file=os.path.join(self.feature_path, data_video_name, 'feature.npy'))
            return self.feature, data_video_name

    def __len__(self):
        return len(self.trainlist)


