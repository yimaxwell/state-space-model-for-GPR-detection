import torch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
import random
from Dataset import *
from TransferModel import *
from TransferModel import Model16
from tqdm import tqdm
import time
import scipy.io as scio
import cv2
import scipy.misc

def setRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed = 1000
setRandomSeed(seed)
cuda = 2
#网络参数
input_size = 256
hidden_size = 512
n_layer = 2

net = Model16(input_size,hidden_size).cuda(cuda)#定义模型
save_path = '/log_merge3/test_result/'#结果保存路径
save_npy_path = save_path + 'npy'

if not os.path.exists(save_path):
        os.mkdir(save_path)

if not os.path.exists(save_npy_path):
        os.mkdir(save_npy_path)

#确定网络参数序号
epoches_per_val = 5
path = '/WorkSpaceP3/zy_test/mamba_test/NetWork/log_merge5/'
pretrained_dict = torch.load(path +'model_pth/autoencoder_model16_bce_29.pth')
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
net_dict.update(pretrained_dict)
net.load_state_dict(pretrained_dict)
torch.cuda.empty_cache()
net.eval()

#读取测试文件
test_data_path = 'Dataset/test_data/'
filelist = os.listdir(test_data_path)
filelist.sort()
for filename in filelist:
    fullname = test_data_path + filename
    #读取mat文件得到网络输入
    test_data = scio.loadmat(fullname)['B_scan_data']
    test_data = test_data.transpose()
    test_data = test_data.astype(float)
    N = test_data.shape[0]
    test_data = cv2.resize(test_data,(256,N))
    test_input_data = (test_data - np.min(test_data))/(np.max(test_data)-np.min(test_data))
    print(test_input_data.shape)
    #输入网络
    test_input_data = torch.tensor(test_input_data).unsqueeze(0).to(torch.float32).cuda(cuda)
    output_data = net(test_input_data).squeeze()

    output_image_data = output_data.cpu().detach().numpy()

    #保存实验结果为图片
    plt.figure()
    plt.imshow(test_data.transpose(),cmap='gray',aspect='auto')
    plt.colorbar()
    plt.savefig(save_path + filename[:-4] + '.png')
    plt.close()
   # plt.imsave(save_path + filename[:-4] + '.png', test_data.transpose(),cmap='gray')#输入
    plt.figure()
    plt.imshow(output_image_data.transpose(),cmap='gray',aspect='auto')
    plt.colorbar()
    plt.show()
    plt.savefig(save_path + filename[:-4] + '_output' + '.png')
    plt.close()
    plt.imsave(save_path + filename[:-4] + '_output2' + '.png', output_image_data.transpose(), cmap='gray')#输出
    #保存实验结果为npy文件
    np.save(save_path + 'npy/input/' + filename[:-4] + '.npy',test_data)
    np.save(save_path + 'npy/output/' + filename[:-4] + '_output' + '.npy',output_image_data)