import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from sklearn.preprocessing import normalize
from utils import euclidean_proj_simplex,D
from numba import jit

maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=0.0


def computeAlpha(Xs,N,loss,rk):    
    #return np.array([1.0/(np.prod(Xs[v].shape)) for v in range(len(Xs))])
    return np.array([1.0 for v in range(len(Xs))])

# Proj gradient update for min_U,b f(U,b)+etaU ||U||_F^2 s.t. U^(r)\in sU*\Delta_1: gradf=\Nabla_{U,b}f(U,b)
# Proj_{sU}([U,b]-step*gradf-step*etaU*[U,0])
@jit(nogil=True,cache=True)
def computeFactorUpdateSimplex(Ub,step,gradf,sU,bias):
    rk = Ub.shape[1]-bias    
    Ub_new = Ub-step*gradf  
    if (not(sU is None)):
        for j in range(rk):
            Ub_new[:,j] = euclidean_proj_simplex(Ub_new[:,j],sU)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)
        
    if bias:
        Ub_new[:,-1] = np.clip(Ub_new[:,-1],minb,None)

    Gt = (Ub-Ub_new)
    Gt2 = la.norm(Gt)**2
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

def FUbk(Ubk,Vbk,bk,Xk,lossk,alpha=1,g=1,Vbk_sum=None,bk_sum=None,eta=0):
    # Vbk,Xk, lossk, alpha are lists 
    # Xv=VvU.T 
    # return f(Ubk) as numpy list and gradF(Ubk)
    
    N=Ubk.shape[0]
    nkv=len(Xk)            
    if eta is None:
        eta=0
        
    if eta>0: f=np.array([0.0 for v in range(nkv+1)]) 
    else: f=np.array([0.0 for v in range(nkv)]) 
    if (not(isinstance(alpha,list)) and not(isinstance(alpha,np.ndarray))):
        alpha=[alpha for v in range(nkv)]  
    if (not(isinstance(lossk,list)) and not(isinstance(lossk,np.ndarray))):
        lossk=[lossk for v in range(nkv)]  
    #if (not(isinstance(Vbk_sum,list))):
    #    Vbk_sum=[Vbk_sum for v in range(nkv)]    
    #if (not(isinstance(bk_sum,list))):
    #    bk_sum=[bk_sum for v in range(nkv)]    
    
    
    if g: 
        gradF=np.zeros(Ubk.shape)    
    for j in range(N):        
        for v in range(nkv):            
            dout = D[lossk[v]](w=Ubk[j,:],A=Vbk[v],x=Xk[v].getcol(j),b=bk[v],g=g,A_sum=Vbk_sum[v],b_sum=bk_sum[v])
            if g:
                ff,gg=dout
                f[v]=f[v]+alpha[v]*ff
                gradF[j,:]=gradF[j,:]+alpha[v]*gg
            else:
                f[v]=f[v]+alpha[v]*dout
    
    if eta>0: f[nkv]=f[nkv]+0.5*eta*la.norm(Ubk)**2 
    f=f.clip(max=maxval)
    if g:
        gradF=gradF+eta*Ubk
        return f, gradF
    return f