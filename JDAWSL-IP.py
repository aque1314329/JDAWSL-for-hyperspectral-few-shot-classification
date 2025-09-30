import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import models
import scipy.io

dataset_name = "IP"
np.float = np.float32
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=120)
parser.add_argument("-c", "--src_input_dim", type=int, default=128)
parser.add_argument("-d", "--tar_input_dim", type=int, default=200)  # SA=204; IP=200 Pa=QY=176
parser.add_argument("-n", "--n_dim", type=int, default=100)
parser.add_argument("-w", "--class_num", type=int, default=16)
parser.add_argument("-s", "--shot_num_per_class", type=int, default=1)
parser.add_argument("-b", "--query_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=20000)
parser.add_argument("-t", "--test_episode", type=int, default=600)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
# target
parser.add_argument("-m", "--test_class_num", type=int, default=16)
parser.add_argument("-z", "--test_lsample_num_per_class", type=int, default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num  # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class  # the number of labeled samples per class 5 4 3 2 1

utils.same_seeds(0)


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')


_init_()
# load source domain data set
with open(os.path.join('datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']  # (77592, 9, 9, 128)
labels_train = source_imdb['Labels']  # 77592
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,18]
print(keys_all_train)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data)  # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)  # (77592, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))  # (9, 9, 100, 77592)
print(source_imdb['data'].shape)  # (77592, 9, 9, 100)
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
# load target domain data set

test_data = 'datasets/IP/indian_pines_corrected.mat'
test_label = 'datasets/IP/indian_pines_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G)  # (10249,) (10249,)
    # print(Row)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}  # Data Augmentation
    m = int(np.max(G))  # 9
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[
                                                                                    RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False,
                                               num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, \
        class_num=class_num, shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    train_datas, train_labels = next(iter(train_loader))
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)  # size of train datas: torch.Size([45, 103, 9, 9])

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain


# 注意力机制
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import functional as F


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * H * W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()  # [16,60,9,9,1]
        # print(x.size())
        proj_query = x.view(m_batchsize, C, -1)  # proj_query[16,60,81]
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # proj_key[16,81,60] # 形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)  # proj_key[16,60,60]
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # torch.max 函数返回两个元素：最大值和最大值的索引,energy_new[16,60,60]
        attention = self.softmax(energy_new)  # attention[16,60,60]
        proj_value = x.view(m_batchsize, C, -1)  # proj_value[16,60,81]

        out = torch.bmm(attention, proj_value)  # out [16,60,81]=torch.bmm{[16,60,60],[16,60,81]}
        out = out.view(m_batchsize, C, height, width, channle)  # [16,60,9,9,1]
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma * out + x  # C*H*W  # [16,60,9,9,1]
        return out


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        x = x.squeeze(-1)  # x:[16,60,9,9,1]->x:[16,60,9,9]
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).views(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).views(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).views(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.views(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()  # x:[16,60,9,9]
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2,
                                                                                      1)  # [16,7,9,9]->[16,7,81]->[16,81,7]

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # [16,7,9,9]->[16,7,81]
        energy = torch.bmm(proj_query, proj_key)  # energe[16,81,81]
        attention = self.softmax(energy)  # attention[16,81,81]
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # proj_value[16,60,81]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # out[16,60,81]
        out = out.view(m_batchsize, C, height, width)  # out[16,60,9,9]

        out = (self.gamma * out + x).unsqueeze(-1)
        return out


class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()

    # Also see https://arxiv.org/abs/1606.08415
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DBDA_network_MISH(nn.Module):
    def __init__(self, band, classes):
        super(DBDA_network_MISH, self).__init__()

        # spectral branch
        self.name = 'DBDA_MISH'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new()
            # swish()
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            # gelu_new()
            # swish()
            mish()
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_spatial = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(120, 2)  # ,
            # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

        # fc = Dense(classes, activation='softmax', name='output1',
        #           kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))

    def forward(self, X):  # X[9,1,100,9,9] -> X[9,1,9,9,100]
        # spectral

        X = X.unsqueeze(1)
        X = X.permute(0, 1, 4, 3, 2)
        x11 = self.conv11(X)  # X11[16,24,9,9,100]
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)  # X12[16,24,9,9,97]
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)  # X13 [16,36,9,9,97]
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)  # X13 [16,12,9,9,97]
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)  # X13 [16,48,9,9,97]
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)  # X13 [16,48,9,9,97]

        x15 = torch.cat((x11, x12, x13, x14), dim=1)  # X15 [16,60,9,9,97]
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)  # X16 [16,60,9,9,1]
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)  # X1 [16,60,9,9,1]
        x1 = torch.mul(x1, x16)  # X1 [16,60,9,9,1]

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)  # X21 [16,24,9,9,1]
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)  # X22 [16,12,9,9,1]

        x23 = torch.cat((x21, x22), dim=1)  # X23 [16,36,9,9,1]
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)  # X23 [16,12,9,9,1]

        x24 = torch.cat((x21, x22, x23), dim=1)  # x24 [16,48,9,9,1]
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)  # x24 [16,12,9,9,1]

        x25 = torch.cat((x21, x22, x23, x24), dim=1)  # x25:[16,60,9,9,1]
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)  # x2:[16,60,9,9,1]
        x2 = torch.mul(x2, x25)  # x2:[16,60,9,9,1]

        # model1
        x1 = self.batch_norm_spectral(x1)  # x1:[16,60,9,9,1]
        x1 = self.global_pooling(x1)  # x1:[16,60,1,1,1]
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.batch_norm_spatial(x2)  # x2:[16,60,9,9,1]
        x2 = self.global_pooling(x2)  # x2:[16,60,1,1,1]
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)  # x_pre[16,120]
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.views(x_pre.shape[0], -1)
        # zhenshi domain_out = self.full_connection(x_pre)  # output[16,16]

        return x_pre


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = DBDA_network_MISH(100, 9)
        self.final_feat_dim = FEATURE_DIM  # 160
        #         self.bn = nn.BatchNorm1d(self.final_feat_dim)
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)  # SRC_INPUT_DIMENSION=128,N_DIMENSION=100
        self.fc2 = nn.Linear(self.final_feat_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, domain='source'):  # x

        if domain == 'target':
            x = self.target_mapping(x)  # (45,100,9,9)
        elif domain == 'source':
            x = self.source_mapping(x)  # (9, 100, 9, 9)
        # print(x.shape)#torch.Size([45, 100, 9, 9])
        feature = self.feature_encoder(x)  # feature(9, 160) x(9,100,9,9)
        domain_out = self.fc2(feature)
        domain_out = self.sigmoid(domain_out)
        # print((feature.shape))
        output = self.classifier(feature)  # output[9,9]
        return feature, domain_out, output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()


def euclidean_metric(a, b):  # 计算支持集和原型特征之间的欧氏距离
    n = a.shape[0]  # 171
    m = b.shape[0]  # 9
    a = a.unsqueeze(1).expand(n, m, -1)  # a[171,9,160]
    b = b.unsqueeze(0).expand(n, m, -1)  # b[171,9,160]
    logits = -((a - b) ** 2).sum(dim=2)  # logits[171,9]
    return logits


# run 10 times
# 超参数设置
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []

best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

seeds = [1309]

class Domain_Occ_loss(nn.Module):
    def __init__(self):
        super(Domain_Occ_loss, self).__init__()

    def forward(self, p1, p2, p3):  # (64,1)  (64,1)

        loss = - torch.mean(torch.log(p1 + 1e-6))
        loss -= torch.mean(torch.log(p2 + 1e-6))
        loss -= torch.mean(torch.log(p3 + 1e-6))

        return loss


# Adaptive Learner
class lossWeight(nn.Module):
    def __init__(self):
        super(lossWeight, self).__init__()
        self.w = nn.Parameter(torch.tensor([4.0, 4.0]), requires_grad=True)

    def forward(self, x1, x2, run=1):
        x = (self.w[0] * x1 + self.w[1] * x2) / (self.w[0] + self.w[1])
        if run % 500 == 0:
            print("w[0]:", self.w[0].item() / (self.w[0].item() + self.w[1].item()), "w[1]:",
                  self.w[1].item() / (self.w[0].item() + self.w[1].item()))
        return x


if __name__ == '__main__':
    trainFlag = False
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from matplotlib.colors import ListedColormap
    if not trainFlag:
        net = Network().cuda()
        net.load_state_dict(torch.load('checkpoints/IP/JDAWSL_IP_0_seeds_1309'))
        rgb_values = [
            [0.0, 0.25, 0.5],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.2],
            [1.0, 0.25, 1.0],
            [0.0, 0.56, 1.0],
            [0.0, 0.85, 1.0],
            [0.59, 1.0, 0.9],
            [0.31, 1.0, 0.69],
            [0.0, 1.0, 0.0],
            [0.82, 1.0, 0.18],
            [1.0, 0.93, 0.0],
            [1.0, 0.68, 0.0],
            [1.0, 0.0, 0.4],
            [1.0, 0.35, 0.0],
            [0.85, 0.0, 0.0],
            [0.5, 0.0, 0.5]
        ]
        custom_cmap = ListedColormap(rgb_values)
        train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
            Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
            shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
        count = 0
        for test_datas, test_labels in test_loader:
            batch_size = test_labels.shape[0]
            train_datas, train_labels = next(iter(train_loader))
            train_features, _, _ = net(Variable(train_datas).cuda(), domain='target')  # (45, 160)

            max_value = train_features.max()
            min_value = train_features.min()
            test_ouput, _, _ = net(Variable(test_datas).cuda(), domain='target')  # (100, 160)
            test_ouput = (test_ouput - min_value) * 1.0 / (max_value - min_value)
            test_ouput = test_ouput.cpu().detach().numpy()
            if count == 0:
                count = 1
                target_features = test_ouput
                y_test = test_labels
            else:
                target_features = np.concatenate((target_features, test_ouput))
                y_test = np.concatenate((y_test, test_labels))

        print(target_features.shape)
        # 创建t-SNE对象，降维到2维
        tsne = TSNE(n_components=2, random_state=42)

        # 使用t-SNE算法降维
        X_tsne = tsne.fit_transform(target_features)
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_tsne_scaled = scaler.fit_transform(X_tsne)
        # 绘制t-SNE图
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_tsne_scaled[:, 0], X_tsne_scaled[:, 1], c=y_test, cmap=custom_cmap)
        plt.xlim(-1.25, 1.25)  # 设置横坐标范围
        plt.ylim(-1.25, 1.25)  # 明设置纵坐标范围

        # 自定义图例
        from matplotlib.lines import Line2D
        # 创建圆形图例元素
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(i + 1),
                   markerfacecolor=rgb_values[i], markersize=10, markeredgecolor='k')
            for i in range(len(rgb_values))
        ]

        plt.legend(handles=legend_elements, title="Class", bbox_to_anchor=(1.0, 1.0), loc='upper left')

        plt.tight_layout()
        plt.show()
    if trainFlag:
        for iDataSet in range(nDataSet):
            loss_Weight = lossWeight()
            # load target domain data for training and testing
            np.random.seed(seeds[iDataSet])
            # train_dataset: 45(目标集的训练集) test_dataset: 42731(目标集的测试集) target_loader:1800(目标集数据增强之后的训练集)
            train_loader, test_loader, target_da_metatrain_data, target_loader, G, RandPerm, Row, Column, nTrain = get_target_dataset(
                Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth, class_num=TEST_CLASS_NUM,
                shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
            # model
            feature_encoder = Network()
            domain_classifier = models.DomainClassifier()
            random_layer = models.RandomLayer([args.feature_dim, args.class_num], 1024)
            DSH_loss = Domain_Occ_loss().cuda()

            feature_encoder.apply(weights_init)
            domain_classifier.apply(weights_init)

            feature_encoder.cuda()
            domain_classifier.cuda()
            random_layer.cuda()  # Random layer

            feature_encoder.train()
            domain_classifier.train()
            # optimizer
            feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
            domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
            weight_optim = torch.optim.Adam(loss_Weight.parameters(), lr=args.learning_rate)

            print("Training...")
            now_best_predict_all = []
            last_accuracy = 0.0
            best_episdoe = 0
            train_loss = []
            test_acc = []
            running_D_loss, running_F_loss = 0.0, 0.0
            running_label_loss = 0
            running_domain_loss = 0
            total_hit, total_num = 0.0, 0.0
            test_acc_list = []
            # source_dataset：77592
            source_iter = iter(source_loader)  # 源域
            target_iter = iter(target_loader)  # 数据增强之后的训练集
            len_dataloader = min(len(source_loader), len(target_loader))
            train_start = time.time()
            for episode in range(5000):  # EPISODE = 10000
                # get domain adaptation data from  source domain and target domain
                try:
                    source_data, source_label = next(
                        source_iter)  # 全部源域数据 len(source_iter)=67, source_data[128,128,9,9]
                except Exception as err:
                    source_iter = iter(source_loader)
                    source_data, source_label = next(source_iter)

                try:
                    target_data, target_label = next(
                        target_iter)  # len(target_iter)=15 数据增强之后的目标域 target_data[128,103,9,9]
                except Exception as err:
                    target_iter = iter(target_loader)
                    target_data, target_label = next(target_iter)

                # source domain few-shot + domain adaptation
                if episode % 2 == 0:
                    '''Few-shot claification for source domain data set'''
                    # get few-shot classification samples
                    # metatrain_data:字典data
                    task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
                    # 取源域的支持集和查询集   metatrain_data=支持集(每类1个，共19个)+查询集(每类19个,共171个)
                    support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS,
                                                                    split="train",
                                                                    shuffle=False)
                    query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                                  shuffle=True)

                    # sample datas
                    supports, support_labels = next(iter(support_dataloader))  # (9, 128, 9, 9)

                    querys, query_labels = next(iter(query_dataloader))  # (171,128,9,9) 19*9=171

                    # calculate features   ------  support_features[9,160]  support_outputs[9,9]  query_features[171,160]  target_outputs[171,9]
                    support_features, supports_dom, support_outputs = feature_encoder(
                        supports.cuda())  # torch.Size([409, 32, 7, 3, 3])
                    query_features, query_dom, query_outputs = feature_encoder(
                        querys.cuda())  # torch.Size([409, 32, 7, 3, 3])
                    target_features, target_dom, target_outputs = feature_encoder(target_data.cuda(),
                                                                                  domain='target')  # torch.Size([409, 32, 7, 3, 3])
                    # target_features[128,160] target_outputs[128,9] target_data[128,103,9,9]
                    # Prototype network
                    if SHOT_NUM_PER_CLASS > 1:
                        support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(
                            dim=1)  # (9, 160)
                    else:
                        support_proto = support_features

                    # fsl_loss  query_features[171,160] support_proto[9,160],logits[171,9]
                    logits = euclidean_metric(query_features, support_proto)
                    f_loss = crossEntropy(logits, query_labels.long().cuda())

                    '''domain adaptation'''
                    # calculate domain adaptation loss   features[308(171+9+128),160]  outputs:[308,9]
                    features = torch.cat([support_features, query_features, target_features], dim=0)
                    outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)  # [308,9]
                    softmax_output = nn.Softmax(dim=1)(outputs)  # [308,9]

                    # set label: source 1; target 0   domain_label[308,1]
                    domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).cuda()
                    domain_label[:supports.shape[0] + querys.shape[0]] = 1  # torch.Size([225=9*20+9*4, 100, 9, 9])
                    # randomlayer_out[308, 1024]  features[308,160],softmax_output[308,9]
                    randomlayer_out = random_layer.forward(
                        [features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])
                    # domain_criterion[308, 1]

                    domain_logits = domain_classifier(randomlayer_out, episode)
                    domain_loss = domain_criterion(domain_logits, domain_label)
                    domain_similar_loss = DSH_loss(supports_dom, query_dom, target_dom)
                    #############################总损失#################################
                    loss = f_loss + loss_Weight(domain_loss, domain_similar_loss, run=episode)
                    # loss = f_loss
                    # Update parameters
                    feature_encoder.zero_grad()
                    domain_classifier.zero_grad()
                    weight_optim.zero_grad()

                    loss.backward()

                    feature_encoder_optim.step()
                    domain_classifier_optim.step()
                    weight_optim.step()

                    total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                    total_num += querys.shape[0]
                # target domain few-shot + domain adaptation
                else:
                    '''Few-shot classification for target domain data set'''
                    # get few-shot classification samples
                    task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,
                                      QUERY_NUM_PER_CLASS)  # 5， 1，15
                    #  使用数据增强后的目标域->目标域的支持集（每类一个样本）
                    support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS,
                                                                    split="train",
                                                                    shuffle=False)
                    # 使用数据增强后的目标域->目标域的查询集（每类19个样本）
                    query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test",
                                                                  shuffle=True)

                    # sample datas  supports[9,103,9,9] querys[171,103,9,9]
                    supports, support_labels = next(iter(support_dataloader))  # (5, 100, 9, 9)
                    querys, query_labels = next(iter(query_dataloader))  # (75,100,9,9)

                    # calculate features  support_features[9,160]  query_features[171,160]  source_features[128,160]
                    support_features, supports_dom, support_outputs = feature_encoder(supports.cuda(),
                                                                                      domain='target')  # torch.Size([409, 32, 7, 3, 3])
                    query_features, query_dom, query_outputs = feature_encoder(querys.cuda(),
                                                                               domain='target')  # torch.Size([409, 32, 7, 3, 3])
                    source_features, source_dom, source_outputs = feature_encoder(
                        source_data.cuda())  # torch.Size([409, 32, 7, 3, 3])

                    # Prototype network
                    if SHOT_NUM_PER_CLASS > 1:
                        support_proto = support_features.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(
                            dim=1)  # (9, 160)
                    else:
                        support_proto = support_features

                    # fsl_loss    欧氏距离logits[171,9]
                    logits = euclidean_metric(query_features, support_proto)
                    ############# 1.交叉熵损失 #############
                    f_loss = crossEntropy(logits, query_labels.long().cuda())

                    '''domain adaptation'''
                    features = torch.cat([support_features, query_features, source_features], dim=0)
                    outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
                    softmax_output = nn.Softmax(dim=1)(outputs)

                    domain_label = torch.zeros(
                        [supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).cuda()
                    domain_label[supports.shape[0] + querys.shape[0]:] = 1

                    randomlayer_out = random_layer.forward(
                        [features, softmax_output])  # torch.Size([225, 1024=32*7*3*3])

                    domain_logits = domain_classifier(randomlayer_out, episode)  # , label_logits
                    ############# 2.域适应损失 #############

                    domain_loss = domain_criterion(domain_logits, domain_label)
                    domain_similar_loss = DSH_loss(supports_dom, query_dom, source_dom)
                    #############################总损失#################################
                    loss = f_loss + loss_Weight(domain_loss, domain_similar_loss, run=episode)
                    # loss = f_loss
                    # Update parameters
                    feature_encoder.zero_grad()
                    domain_classifier.zero_grad()
                    weight_optim.zero_grad()

                    loss.backward()
                    feature_encoder_optim.step()
                    domain_classifier_optim.step()

                    weight_optim.step()
                    total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                    total_num += querys.shape[0]

                if (episode + 1) % 100 == 0:  # display
                    display_time = time.time()
                    train_loss.append(loss.item())
                    print(
                        'episode {:>3d}: time: {:6.4f}, domain loss: {:6.4f}, domain_similar_loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(
                            episode + 1, \
                            display_time - train_start,
                            domain_loss.item(),
                            f_loss.item(),
                            domain_similar_loss.item(),
                            # 0.02 * contrastive_loss_t.item(),
                            total_hit / total_num,
                            loss.item()))

                if (episode + 1) % 200 == 0 or episode == 0:  # 开始测试
                    # test
                    print("Testing ...")
                    train_end = time.time()
                    feature_encoder.eval()
                    total_rewards = 0
                    counter = 0
                    accuracies = []
                    predict = np.array([], dtype=np.int64)
                    labels = np.array([], dtype=np.int64)

                    # train_loader 目标集的训练集：每类五个样本
                    train_datas, train_labels = next(iter(train_loader))
                    train_features, _, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

                    max_value = train_features.max()  # 89.67885
                    min_value = train_features.min()  # -57.92479
                    print(max_value.item())
                    print(min_value.item())
                    # 归一化
                    train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                    KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                    KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)  # .cpu().detach().numpy()
                    for test_datas, test_labels in test_loader:
                        batch_size = test_labels.shape[0]

                        test_features, _, _ = feature_encoder(Variable(test_datas).cuda(),
                                                              domain='target')  # (100, 160)
                        test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                        predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                        test_labels = test_labels.numpy()
                        rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                        total_rewards += np.sum(rewards)
                        counter += batch_size

                        predict = np.append(predict, predict_labels)
                        labels = np.append(labels, test_labels)

                        accuracy = total_rewards / 1.0 / counter  #
                        accuracies.append(accuracy)

                    test_accuracy = 100. * total_rewards / len(test_loader.dataset)

                    print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                                   100. * total_rewards / len(test_loader.dataset)))
                    test_end = time.time()

                    # Training mode
                    feature_encoder.train()
                    if test_accuracy > last_accuracy:
                        now_best_predict_all = predict
                        # save networks
                        # torch.save(feature_encoder.state_dict(),
                        #            str("checkpoints/SA/A=0.3---" + "SA_" + str(iDataSet) + "iter_" + str(
                        #                TEST_LSAMPLE_NUM_PER_CLASS) + "episode_" + str(episode + 1) + "shot.pkl"))
                        # print("save networks for episode:", episode + 1)
                        last_accuracy = test_accuracy
                        best_episdoe = episode

                        acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                        OA = acc
                        C = metrics.confusion_matrix(labels, predict)
                        A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float32)

                        k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                    print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            torch.save(feature_encoder.state_dict(),
                       str("checkpoints/{}/JDAWSL" + "_" + str(dataset_name) + "_" + str(iDataSet) + "_seeds_" + str(
                           seeds[iDataSet]) + "_shot_" + str(TEST_LSAMPLE_NUM_PER_CLASS) + "_accuracy_" + str(
                           last_accuracy) + ".pkl").format(dataset_name))
            if test_accuracy > best_acc_all:
                best_predict_all = now_best_predict_all
                best_acc_all = test_accuracy
                best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
                    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = \
                        best_predict_all[
                            i] + 1

            scipy.io.savemat(
                './classificationMap/{}/shot{}_seed{}_JDAWSL_{}_{}.mat'.format(dataset_name, TEST_LSAMPLE_NUM_PER_CLASS,
                                                                              seeds[iDataSet], dataset_name,
                                                                              acc[iDataSet]),
                {'pre_gt': G[4:-4, 4:-4]})

            print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
            print('***********************************************************************************')

        AA = np.mean(A, 1)

        AAMean = np.mean(AA, 0)
        AAStd = np.std(AA)

        AMean = np.mean(A, 0)
        AStd = np.std(A, 0)

        OAMean = np.mean(acc)
        OAStd = np.std(acc)

        kMean = np.mean(k)
        kStd = np.std(k)
        print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
        print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
        print("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
        print("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
        print("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
        print("accuracy for each class: ")
        for i in range(CLASS_NUM):
            print("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

        best_iDataset = 0
        for i in range(len(acc)):
            print('{}:{}'.format(i, acc[i]))
            if acc[i] > acc[best_iDataset]:
                best_iDataset = i
        print('best acc all={}'.format(acc[best_iDataset]))





