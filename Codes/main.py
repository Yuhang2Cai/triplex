'''PhyCRNet for solving spatiotemporal PDEs'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
from torch.utils.data import DataLoader, TensorDataset

from PhyConvNet import PhyConvNet
from train_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(66)
np.random.seed(66)
torch.set_default_dtype(torch.float32)

Nx= 64
Ny =64
BHPmat = scio.loadmat('BHP_full.mat')
BHP_vec = torch.tensor(BHPmat['BHP_full'], dtype=torch.float32).cuda() 

# print(BHP_vec)
# Rate  = np.array([[100]]) # [STB/day]
# Rate_vec =      torch.tensor(Rate, dtype=torch.float32).repeat(1,300).cuda()
Rate = scio.loadmat('Qinj_full.mat')

Rate_vec = torch.tensor(Rate['Qinj_full'], dtype=torch.float32).cuda()
print("rate_vec.size" + str(Rate_vec.shape))
# print(Rate_vec)

TRUE_PERM = scio.loadmat('/home/lxchen/data/cyh/PICNN-twophaseporousflow-main/Codes/system/TRUE_PERM_64by64.mat')
Perm = torch.tensor(TRUE_PERM['TRUE_PERM'], dtype=torch.float32).cuda()
print("Perm.size" + str(Perm.shape))
# Perm = torch.tensor(0.10*np.ones((Nx, Ny)), dtype=torch.float32).cuda()

if __name__ == '__main__':
    print(os.getcwd())
    ######### download the ground truth data ############
#     data_dir = '/content/PhyCRNet-main-1phaseflow/Datasets/data/2dBurgers/burgers_1501x2x128x128.mat'    
    data_dir_p = '/home/lxchen/data/cyh/PICNN-twophaseporousflow-main/Datasets/data/twophaseflow/pressure_101x1x64x64.mat'
    data_dir_sw = '/home/lxchen/data/cyh/PICNN-twophaseporousflow-main/Datasets/data/twophaseflow/saturation_101x1x64x64.mat'
    data_p = scio.loadmat(data_dir_p)
    data_sw = scio.loadmat(data_dir_sw)
    # uv = data['uv'] # [t,c,h,w]  
    p = data_p['Psim'] # [t,c,h,w]  
    sw = data_sw['Swsim'] # [t,c,h,w] 

    # initial conidtion
    p0 = torch.tensor(p[0:1,...], dtype=torch.float32).cuda()
    # p0 = p0/3000
    sw0 = torch.tensor(sw[0:1,...], dtype=torch.float32).cuda()
    print("p0.shape: "+str(p0.shape))
    print("sw0.shape: "+str(sw0.shape))
    steps_net = 50
    dt = 2
    
    # time map  np.array(range(1,301,dt))
    steps_sim = 1
    
    Tmap = torch.tensor(np.zeros((steps_net,1,64,64)), dtype=torch.float32).cuda()
    for k in range(steps_net):
        Tmap[k:k+1,...] = torch.tensor(np.ones((1,1,64,64))*(k+1)*dt/steps_sim, dtype=torch.float32).cuda()
        # print("Tmap[k:k+1,...].size"+ str(Tmap[k:k+1,...].shape))
# #     T =
    # source BHP 
    BHP = np.zeros((steps_net,1,64,64))
    source_BHP = torch.tensor(BHP, dtype=torch.float32).cuda()
    source_BHP[:,0, 49, 49]=BHP_vec[0,:steps_net*dt:dt]    # Pi =3000
    source_BHP[:,0, 12, 12] = BHP_vec[1,:steps_net*dt:dt]
    
    rate = np.zeros((steps_net,1,64,64))
    Qinj = torch.tensor(rate, dtype=torch.float32).cuda()
    Qinj[:, 0,31,31] = Rate_vec[0,:steps_net*dt:dt]   # Qmax =1500
    Qinj[:, 0,12,49] = Rate_vec[1,:steps_net*dt:dt]
    Qinj[:, 0,49,12] = Rate_vec[2,:steps_net*dt:dt]

    # P0 = p0.repeat(time_sim, 1, 1, 1)
    # SW0 = sw0.repeat(time_sim, 1, 1, 1)
    # print(BHP_vec[1,:steps_net])
    # inputs = torch.cat((Qinj, source_BHP, P0, SW0), dim=1)    # concat time map and control 
    
    
    # dataset = TensorDataset(inputs)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    ################# build the model #####################
    # time_batch_size = 300
    # steps = time_batch_size
    # effective_step = list(range(0, steps))
    # num_time_batch = int(time_steps / time_batch_size)
    np.ones((steps_net-1, 1))
    n_iters_adam = 30000
    lr_adam = 0.01 #1e-3 

    fig_save_path = '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Datasets/figures/'  

    # model = PhyConvNet(latent_size = 200, dt = dt, time_sim = steps_net, 
    #     input_channels = 4,  
    #     hidden_channels = [16, 32, 64, 128, 64, 32, 16], 
    #     input_kernel_size = [4, 4, 4, 4, 4, 4, 4], 
    #     input_stride = [2, 2, 2, 2, 2, 2, 2], 
    #     input_padding = [1, 1, 1, 1, 1, 1, 1],  
    #     num_layers=[4,3,1], 
    #     upscale_factor=8).cuda()


    start = time.time()
    train_loss, outputs = train(source_BHP, Qinj, Tmap, p0, sw0, n_iters_adam,
        lr_adam, dt, Rate_vec[:,:steps_net*dt:dt] , BHP_vec[:,:steps_net*dt:dt] , Perm, steps_net, steps_sim)
    end = time.time()
    
    np.save('/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/model/train_loss', train_loss)  
    print('The training time is: ', (end-start))

    """主程序函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 数据加载和预处理
    print("1. 加载和预处理数据...")
    data = pd.read_csv('combined_all_t_p.csv')

    # 修改：去掉最后一列'imotor'，只保留前三列作为输入
    X = data.drop(['pOut', 'iMotor'], axis=1)  # 去掉目标列和不需要的列
    Y = data['pOut']
    #物理参数
    V = 2 * np.exp(-4),
    bulk_modulus_model = 'const',
    air_dissolution_model = 'off',
    rho_L_atm = 851.6,
    beta_L_atm = 1.46696e+03,
    beta_gain = 0.2,
    air_fraction = 0.005,
    rho_g_atm = 1.225,
    polytropic_index = 1.0,
    p_atm = 0.101325,
    p_crit = 3,
    p_min = 1

    start = time.time()
    train_loss, outputs = train(source_BHP, Qinj, Tmap, p0, sw0, n_iters_adam,
                                lr_adam, dt, Rate_vec[:, :steps_net * dt:dt], BHP_vec[:, :steps_net * dt:dt], Perm,
                                steps_net, steps_sim)
    end = time.time()

    np.save(
        '/scratch/user/jungangc/PICNN-2phase/PICNN-2phaseflow-constBHP-64by64-heter-TransferLearning-50stepsby2-final/Codes/model/train_loss',
        train_loss)
    print('The training time is: ', (end - start))
















