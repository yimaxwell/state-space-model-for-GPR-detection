import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
import random
from Dataset import my_collate,GetData,LenMatchBatchSampler,calculate_batch_len,GetPaddedData,MyBatchSampler
from TransferModel import *
from tqdm import tqdm
import time

def setRandomSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 1000
setRandomSeed(seed)
cuda = 2


train_datapath = 'DataSet/DataSet_merge3/merge/images/rawtraining/'
train_labelpath = 'DataSet/DataSet_merge3/merge/annotations/rawtraining/'
val_datapath = 'DataSet/DataSet_merge3/merge/images/rawvalidation/'
val_labelpath = 'DataSet/DataSet_merge3/merge/annotations/rawvalidation/'
save_path = '/log_merge3'

epoches_per_val = 5

batch_size = 32

train_dataset = GetData(train_datapath,train_labelpath)
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=False)
batch_sampler = LenMatchBatchSampler(train_dataset,batch_size=batch_size,drop_last=False)
batch_idx, batch_len = calculate_batch_len(train_dataset,batch_sampler)
padded_dataset = GetPaddedData(train_dataset, batch_idx, batch_len,200)
my_batch_sampler = MyBatchSampler(batch_idx)
train_dataloader = torch.utils.data.DataLoader(padded_dataset,batch_sampler=my_batch_sampler,collate_fn=my_collate)

val_dataset = GetData(val_datapath,val_labelpath)
val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=1,shuffle=False)

train_num = len(train_dataset)
val_num = len(val_dataset)

encoder_hidden_size_list = [512,256,256,128]
decoder_hidden_size_list = [128,256,256,512]

input_size = 256
hidden_size = 512

net = Model16(input_size,hidden_size).cuda(cuda)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[200], gamma=0.1)  # 200epoch 学习率衰减为0.01
loss_function = torch.nn.BCELoss()
train_loss=[]
val_loss=[]
epochs = 30
output_list = []
torch.cuda.empty_cache()

for epoch in range(epochs):
    net.train()
    train_bar = tqdm(train_dataloader)
    train_loss_total = 0
    for idx, (data,label,_) in enumerate(train_bar):
        M,N = data.shape[2], data.shape[1]
        train_bar.set_description('epoch: {}/{}, batch_idx: {}/{}'.format(epoch, epochs, idx + 1, train_num))
        data = torch.tensor(data).to(torch.float32).cuda(cuda)
        label = torch.tensor(label).to(torch.float32).cuda(cuda)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_function(output,label)
        train_loss_total += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

    train_loss.append(train_loss_total.item() / (idx + 1))
    print('Train Loss: {}'.format(train_loss_total.item()/(idx+1)))
    scheduler.step()


    if (epoch+1)%epoches_per_val==0 or (epoch+1)==epochs:
        net.eval()
        val_bar = tqdm(val_dataloader)
        val_loss_total = 0
        with torch.no_grad():
            for idx, (data,label) in enumerate(val_bar):
                M, N = data.shape[2], data.shape[1]
                val_bar.set_description('epoch: {}/{}, batch_idx: {}/{}'.format(epoch, epochs, idx + 1, val_num))
                data = torch.tensor(data).to(torch.float32).cuda(cuda)
                label = torch.tensor(label).to(torch.float32).cuda(cuda)
                optimizer.zero_grad()
                output = net(data)
                if epoch+1==epochs:
                    output_list.append(output.cpu().squeeze().detach().numpy().transpose())
                loss = loss_function(output.squeeze(),label.squeeze())
                val_loss_total += loss
                # torch.cuda.empty_cache()
            val_loss.append(val_loss_total.item() / val_num)
            print('Val Loss: {}'.format(val_loss_total.item() / val_num))
        torch.save(net.state_dict(), save_path + '/model_pth/autoencoder_model16_bce_{}.pth'.format(epoch))


np.save(save_path+'/train_loss_model16.npy',train_loss)
np.save(save_path+'/val_loss_model16.npy',val_loss)
torch.save(net.state_dict(), save_path+'/model_pth/autoencoder_model16_bce.pth')

plt.figure()
plt.plot(np.arange(1,epochs+1,step=1),np.array(train_loss))
plt.plot(np.arange(epoches_per_val,epochs-1+epoches_per_val,step=epoches_per_val),np.array(val_loss))
plt.savefig(save_path+'/loss.png')
plt.show()

