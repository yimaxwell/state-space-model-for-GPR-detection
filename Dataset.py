from torch.utils.data import Dataset
import torch
import os
import numpy as np
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Sampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import random
import cv2
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

# from sklearn.decomposition import PCA

def setRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 1000
setRandomSeed(seed)

def my_collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    buff = [item[2] for item in batch]
    return (torch.stack(data), torch.stack(label), torch.stack(buff).squeeze(1))

def calculate_batch_len(dataset,batch_sampler):
    n = len(batch_sampler)
    batch_sampler = iter(batch_sampler)
    batch_len = []
    batch_idx = []
    for i in range(n):
        idx = next(batch_sampler)
        batch_len.append(idx)
        batch_idx.append([dataset[x][0].shape[0] for x in idx])
    return batch_len, batch_idx

class GetData(Dataset):
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.data_list = os.listdir(self.data_path)
        self.label_list = os.listdir(self.label_path)
        self.data_list.sort(key=lambda x:int(x[:-4]))
        self.label_list.sort(key=lambda x:int(x[:-4]))
        # self.data_list.sort()
        # self.label_list.sort()
        self.data = []
        self.label = []
        # self.label = torch.tensor(np.load(label_path)).to(torch.int64)
        #for i in range(len(self.data_list)):
        for filename in self.data_list:
            #data = cv2.imread(self.data_path + '{}.jpg'.format(i))
            #label = cv2.imread(self.label_path + '{}.png'.format(i))
            data = cv2.imread(self.data_path +  filename)
            label = cv2.imread(self.label_path + filename[:-4] + '.png')
            N = data.shape[1]
            label = cv2.resize(label, (N, 256))[:,:,0] 
            label[label>=1] = 1
            label[label<=0] = 0
            # print(data.shape)
            data = cv2.resize(data, (N, 256))[:, :, 0] / 255
            # print(data.shape)
            self.data.append(torch.tensor(np.transpose(data)))
            self.label.append(torch.tensor(np.transpose(label)))

    def __getitem__(self, idx):
        # data = cv2.imread(self.data_path + '{}.jpg'.format(idx))
        # label = cv2.imread(self.label_path + '{}.png'.format(idx))
        # N = data.shape[1]
        # label = cv2.resize(label, (N, 512))[:, :, 0]
        # data = cv2.resize(data, (N, 512))[:, :, 0] / 255
        # return np.transpose(data), np.transpose(label)
        # label = F.one_hot(self.label[idx],num_classes=100)
        return self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.data_list)

class GetPaddedData(Dataset):
    def __init__(self, dataset, batch_idx, batch_len, max_len = None):
        self.dataset = dataset
        self.batch_idx = batch_idx
        self.batch_len = batch_len
        self.N = len(batch_len)
        self.max_len = max_len

        if max_len == None:
            self.max = [max(x) for x in batch_len]
        else:
            self.max = [min(max_len,max(x)) for x in batch_len]

    def __getitem__(self, idx):

        for i in range(self.N):
            try:
                n = self.batch_idx[i].index(idx)
                break
            except:
                pass

        batch_number = self.max[i]

        l = self.dataset[idx][1]
        d = self.dataset[idx][0]
        N = d.shape[0]
        if N>batch_number-2:
            N = batch_number - 2
            l = l[:N,:]
            d = d[:N,:]

        # print(l.shape,batch_number)

        # l = cv2.resize(l,(N,512))[:,:,0]
        # d = cv2.resize(d, (N,512))[:,:,0]/255

        label = torch.zeros((batch_number,256),dtype=int)
        data = torch.zeros((batch_number,256), dtype=float)

        label[:N,:] = l
        data[:N,:] = d

        return data.to(torch.float32), label.to(torch.int32), torch.tensor([N])

    def __len__(self):
        return len(self.dataset)


class MyBatchSampler():

    def __init__(self, batch_idx):
        self.batch_idx = batch_idx

    def __iter__(self):
        for idx in self.batch_idx:
            yield idx

    def __len__(self) -> int:
        return len(self.batch_idx)

class GetAutoEncoderData(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.list = os.listdir(self.data_path)

    def __getitem__(self, idx):
        data = cv2.imread(self.data_path+self.list[idx])
        N = data.shape[1]
        data = cv2.resize(data,(N,512))
        return np.transpose(data[:,:,0]/255)

    def __len__(self):
        return len(self.list)

class LenMatchBatchSampler():
    def __init__(self, dataset, batch_size, drop_last):
        self.batch_size = batch_size
        self.n = len(dataset)
        self.size = np.zeros((self.n,),dtype=int)
        self.drop_last = drop_last
        for i in range(self.n):
            self.size[i] = dataset[i][0].shape[0]
        self.idx = np.argsort(self.size)
        if self.n % self.batch_size == 0:
            self.len = self.n // self.batch_size
        else:
            self.len = self.n // self.batch_size + 1
        if self.drop_last:
            self.len -= 1
            self.n = self.len * self.batch_size
        self.shuffle_batch_idx = list(range(self.len))
        random.shuffle(self.shuffle_batch_idx)

    def __iter__(self):
        for i in range(self.len):
            idx = self.shuffle_batch_idx[i]
            if idx == self.len-1:
                bucket = list(range(idx*self.batch_size,self.n))
            else:
                bucket = list(range(idx*self.batch_size,(idx+1)*self.batch_size))
            random.shuffle(bucket)
            yield list(self.idx[bucket])

    def __len__(self):
        return self.len

if __name__ == '__main__':


    seed = 1000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    data_path = '/WorkSpaceP2/GCL/mamba_test/DataSet/merge/images/rawtraining/'
    label_path = '/WorkSpaceP2/GCL/mamba_test/DataSet/merge/annotations/rawtraining/'
    # label_path = '/WorkSpaceP2/GCL/mamba_test/DataSet/merge/annotations/rawtraining_newlabel/'
    # label_path = '/WorkSpaceP2/GCL/mamba_test/DataSet/merge/training_target_number.npy'

    dataset = GetData(data_path,label_path)
    padded_dataset = GetPaddedData(dataset,[[0,1]],[[200,200]],256)
    d,l,n = padded_dataset[-1]
    # dd,ll = dataset[-2]

    print(d.shape,len(dataset),n)
    # print(l)
    # print(torch.max(d),torch.min(d))
    # print(torch.max(l), torch.min(l))
    # print(num)

    plt.figure()
    plt.imshow(d.detach().numpy(),cmap='gray',aspect='auto')
    plt.show()
    plt.figure()
    plt.imshow(l.detach().numpy(), cmap='gray',aspect='auto')
    plt.show()
    plt.figure()
    plt.plot(d[0,:].detach().numpy())
    plt.plot(l[-1, :].detach().numpy())
    plt.show()

    # model = PCA(n_components=64)
    # Z = model.fit_transform(d.transpose())
    # Z2 = model.fit_transform(dd.transpose())
    # print(Z.shape)
    # print(np.sum(model.explained_variance_ratio_))
    # X = model.inverse_transform(Z)
    # plt.figure()
    # plt.imshow(X, cmap='gray',aspect='auto')
    # plt.show()

