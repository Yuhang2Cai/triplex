import scipy.io as scio
import torch
import numpy as np

def compute_source(Discretization, Mobility, Wells):
    Nt          = Discretization.Nt
    fro         = Mobility.fro
    frw         = Mobility.frw
    Qinj        = Wells.Qinj;
    Pwf_pro     = Wells.Pwf_pro;
    J_pro       = Wells.J_pro
    Qinj_ind    = Wells.Qinj_ind
    Qpro_ind    = Wells.Qpro_ind
    Npro        = Wells.Npro
    tsteps = Discretization.Tsteps
    print(f"Nt (总网格数): {Nt} (标量)")  # 应为标量，如64×64=4096
    # 打印流度相关项形状
    print(f"fro (油相流度项): {fro.shape}")  # 预期 [batch_size, 1, H, W]，如[2,1,64,64]
    print(f"frw (水相流度项): {frw.shape}")  # 预期 [batch_size, 1, H, W]
    # 打印注入井源项形状
    print(f"Qinj (注入率): {Qinj.shape}")  # 预期 [batch_size, 1, H, W]，仅注入井位置非零
    # 打印生产井相关参数形状
    print(f"Pwf_pro (生产井井底压力): {Pwf_pro.shape}")  # 预期 [batch_size, Npro, 1]，如[2,2,1]（2口井）
    print(f"J_pro (生产井井指数项): {J_pro.shape}")  # 预期 [batch_size, Npro, 1]，与生产井数量匹配
    # 打印井位置索引形状
    print(f"Qinj_ind (注入井索引): {Qinj_ind.shape}")  # 预期 [Ninj, 2]，如[3,2]（3口注入井的坐标）
    print(f"Qpro_ind (生产井索引): {Qpro_ind.shape}")  # 预期 [Npro, 2]，如[2,2]（2口生产井的坐标）
    # 打印生产井数量
    print(f"Npro (生产井数量): {Npro} (标量)")  # 应为标量，如2
    print(tsteps)
    #     %================================
    Qo      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    Qw      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    qo      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    qw      = torch.zeros((Nt,tsteps), dtype=torch.float32).cuda()
    
#     zero_mat = torch.zeros((Nt, Nt), dtype=torch.float32)
    

#     % Producer wells
#     %---------------------
    # print(Pwf_pro.size())
    # if tsteps>1:
    Qo[Qpro_ind]    =  torch.unsqueeze(J_pro, 1)*fro[Qpro_ind]#J_pro (生产井井指数项): torch.Size([2, 2, 64])  Qpro_ind (生产井索引): (2,)
    Qw[Qpro_ind]    =  torch.unsqueeze(J_pro, 1)*frw[Qpro_ind]

    qo[Qpro_ind]    =  Pwf_pro*torch.unsqueeze(J_pro, 1)*fro[Qpro_ind]
    qw[Qpro_ind]    =  Pwf_pro*torch.unsqueeze(J_pro, 1)*frw[Qpro_ind]

            

#     %---------------------
#     % Injector wells
#     %---------------------
    qw[Qinj_ind]    = qw[Qinj_ind] + Qinj

    # print('producer:', Qo)
    
    # Qo_lst = []
    # Qw_lst = []
    # qo_lst =[]
    # qw_lst =[]
    
    # for k in range(tsteps):
    diagonal_Qo = torch.diag(Qo.squeeze())
    Qo_lst = diagonal_Qo
    diagonal_Qw = torch.diag(Qw.squeeze())
    Qw_lst = diagonal_Qw
#     %=========================
    Q_o = torch.hstack((Qo_lst, torch.zeros_like(Qo_lst)))
    Q_w = torch.hstack((Qw_lst, torch.zeros_like(Qw_lst)))
    Q   = torch.vstack((Q_o, Q_w))
    # for k in range(tsteps):
    diagonal_qo = qo
    qo_lst = diagonal_qo
    diagonal_qw = qw
    qw_lst = diagonal_qw      
#     %=========================
    q   = torch.vstack((qo_lst, qw_lst))

#     %=========================================
    Wells.Qo     = Qo_lst
    Wells.Qw     = Qw_lst
    Wells.Q     = Q
    Wells.qo     = qo_lst
    Wells.qw     = qw_lst
    Wells.q      = q
    
    return Wells