import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from sklearn.preprocessing import normalize
from utils import euclidean_proj_simplex,D

maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=1e-10

@jit(nogil=True,cache=True)
def computeAlpha(Xs,entV,N,loss,rk):    
    return np.array([1.0 for v in range(len(Xs))])

# Proj gradient update for min_U,b f(U,b)+etaU ||U||_F^2 s.t. U^(r)\in sU*\Delta_1: gradf=\Nabla_{U,b}f(U,b)
# Proj_{sU}([U,b]-step*gradf-step*etaU*[U,0])
@jit(nogil=True,cache=True)
def computeFactorUpdateSiCNMF(Ub,step,gradf,sU,etaU,bias):
    rk = Ub.shape[1]-bias
    
    Ub_new = Ub-step*gradf        
    if (not(etaU is None)):
        Ub_new[:,:rk] = Ub_new[:,:rk]-step*etaU*Ub[:,:rk]
        
    if (not(sU is None)):
        for j in range(rk):
            Ub_new[:,j] = euclidean_proj_simplex(Ub_new[:,j],sU)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)
        
    if bias:
        Ub_new[:,-1] = np.clip(Ub_new[:,-1],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

@jit(nogil=True,cache=True)
def FUbk(Ubk,Vbk,bk,Xk,lossk,alpha=1,g=1,Vbk_sum=None,bk_sum=None):
    # Vbk,Xk, lossk, alpha are lists 
    # Xv=VU.T 
    # return f(Ubk) as numpy list and gradF(Ubk)
    
    N=Ubk.shape[0]
    nkv=len(Xk)        
    f=np.array([0.0 for v in range(nkv)])
    if (not(isinstance(alpha,list))):
        alpha=[alpha for v in range(nkv)]    
    if (not(isinstance(Vbk_sum,list))):
        Vbk_sum=[Vbk_sum for v in range(nkv)]    
    if (not(isinstance(bk_sum,list))):
        bk_sum=[bk_sum for v in range(nkv)]    
        
    if g: 
        gradF=np.zeros(Ubk.shape)    
    for j in range(Ubk.shape[0]):        
        for v in range(nkv):
            dout = D[lossk[v]](w=Ubk[j,:],A=Vbk[v],x=Xk[v].getcol(j),b=bk[v],g=g,A_sum=Vbk_sum[v],b_sum=bk_sum[v])
            if g:
                ff,gg=dout
                f[v]=f[v]+alpha[v]*ff
                gradF[j,:]=gradF[j,:]+alpha[v]*ff
            else:
                fest[v]=fest[v]+alpha[v]*dout
                  
    f=f.clip(max=maxval)
    if g: return f, gradF
    return f