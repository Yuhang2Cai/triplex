import scipy.io as scio
import torch
import numpy as np
# from scipy.sparse import spdiags, csr_matrix

def compute_mob(kr,b,U):
    y   = kr*b/U
    return y

def compute_dmob_dP(kr,b,U,db,dU):
    dydp    = kr*(db*U-dU*b)/(U**2)
    return dydp

def compute_dmob_dSw(dkr,b,U):
    dyds    = dkr*b/U
    return dyds


class struct:
    pass

def compute_mobility(Fluid,Rock):
    bo      = Fluid.bo
    bw      = Fluid.bw
    # dbo     = Fluid.dbo;
    # dbw     = Fluid.dbw;
    Uo      = Fluid.Uo
    Uw      = Fluid.Uw
    # dUo     = Fluid.dUo;
    # dUw     = Fluid.dUw;
    kro     = Rock.kro
    krw     = Rock.krw
    # dkro    = Rock.dkro;
    # dkrw    = Rock.dkrw;
#     type = Discretization.Geom_AvgType;
    print(f"bo (原油体积系数): {bo.shape}")
    print(f"bw (地层水体积系数): {bw.shape}")
    print(f"Uo (原油黏度相关项): {Uo.shape}")
    print(f"Uw (地层水黏度相关项): {Uw.shape}")
    print(f"kro (油相相对渗透率): {kro.shape}")
    print(f"krw (水相相对渗透率): {krw.shape}")
    # %--------------------
    fro     = compute_mob(kro,bo,Uo)
    frw     = compute_mob(krw,bw,Uw)
    print("fro.shape: " + str(torch.tensor(fro).shape))
    print("frw.shape: " + str(torch.tensor(frw).shape))
    # print(fro)
#     %=================================================
    # dfrodP      = compute_dmob_dP(kro,bo,Uo,dbo,dUo);
    # dfrwdP      = compute_dmob_dP(krw,bw,Uw,dbw,dUw);
    # dfrodSw     = compute_dmob_dSw(dkro,bo,Uo);
    # dfrwdSw     = compute_dmob_dSw(dkrw,bw,Uw);    
#     %=================================================
    Mobility = struct()
    Mobility.fro        = fro
    Mobility.frw        = frw
    Mobility.frt        = fro +frw
    # Mobility.dfrodP     = dfrodP;
    # Mobility.dfrwdP     = dfrwdP;
    # Mobility.dfrodSw    = dfrodSw;
    # Mobility.dfrwdSw    = dfrwdSw;
    
    return Mobility