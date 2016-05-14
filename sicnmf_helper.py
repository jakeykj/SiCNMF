import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from sklearn.preprocessing import normalize
from utils import euclidean_proj_simplex,D
from numba import jit


maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=0.0


def computeAlphaNew(Xs,N,loss,eta,rk):  
    #HARDCODED VALUES
    #assuming p_bias=0, w_bias=1 and V=2
    print "alpha computation"
    X=Xs[0].tocsc()
    Upat = np.random.rand(X.shape[0],rk)
    if (not(eta is None) and la.norm(Upat[:,:rk])>eta): Upat = eta*Upat/la.norm(Upat)
    Ub0=np.random.rand(X.shape[1],rk+1); Ub0=Ub0/Ub0.sum(0) 
    f=[np.array([0.0])]*2
    for i in range(5):
        Vb=np.hstack((Upat,np.ones((X.shape[0],1))))
        b=np.zeros((X.shape[0],))
        gfs = {'func':FUbk, 'args':{'Xk':[X.tocsc()],'lossk':loss,'alpha':[1],\
                                    'Vbk':[Vb],'bk':[b],'Vbk_sum':[Vb.sum(0)],'bk_sum':[b.sum()]}}
        nextFactors = {'func':computeFactorUpdate,'args':{'sU':1,'eta':None,'bias':1}}
        Ub0,f[0],_=singleUbUpdate(Ub0,gfs,nextFactors,10,verbose)  

        gfs = {'func':FUbk, 'args':{'Xk':[X.T.tocsc()],'lossk':loss,'alpha':[1],\
                                    'Vbk':[Ub0[:,:rk]],'bk':[Ub0[:,rk]],'Vbk_sum':[Ub0[:,:rk].sum(0)],'bk_sum':[Ub0[:,rk].sum()]}}        
        nextFactors = {'func':computeFactorUpdate,'args':{'sU':None,'eta':eta,'bias':0}}
        Upat,f[0],_=singleUbUpdate(Upat,gfs,nextFactors,10,verbose)  
   
    X=Xs[1].tocsc()
    Upat = np.random.rand(X.shape[0],rk)
    if (not(eta is None) and la.norm(Upat[:,:rk])>eta): Upat = eta*Upat/la.norm(Upat)
    Ub1=np.random.rand(X.shape[1],rk+1); Ub1=Ub1/Ub1.sum(0)     
    for i in range(5):
        Vb=np.hstack((Upat,np.ones((X.shape[0],1))))
        b=np.zeros((X.shape[0],))
        gfs = {'func':FUbk, 'args':{'Xk':[X.tocsc()],'lossk':loss,'alpha':[1],\
                                    'Vbk':[Vb],'bk':[b],'Vbk_sum':[Vb.sum(0)],'bk_sum':[b.sum()]}}
        nextFactors = {'func':computeFactorUpdate,'args':{'sU':1,'eta':None,'bias':1}}
        Ub1,f[1],_=singleUbUpdate(Ub1,gfs,nextFactors,10,verbose)  

        gfs = {'func':FUbk, 'args':{'Xk':[X.T.tocsc()],'lossk':loss,'alpha':[1],\
                                    'Vbk':[Ub1[:,:rk]],'bk':[Ub1[:,rk]],'Vbk_sum':[Ub1[:,:rk].sum(0)],'bk_sum':[Ub1[:,rk].sum()]}}        
        nextFactors = {'func':computeFactorUpdate,'args':{'sU':None,'eta':eta,'bias':0}}
        Upat,f[1],_=singleUbUpdate(Upat,gfs,nextFactors,10,verbose)  
        
    alpha =  np.array([f[1]/f[0],1.0])
    print "end alpha computation", alpha
    return alpha, [Ub0, Ub1]
    #return np.array([1.0 for v in range(len(Xs))])
    
def computeAlpha(Xs,N,loss,rk):  
    #HARDCODED VALUES
    #assuming p_bias=0, w_bias=1 and V=2
    return np.ones((len(Xs),))

# Proj gradient update for min_U,b f(U,b)+etaU ||U||_F^2 s.t. U^(r)\in sU*\Delta_1: gradf=\Nabla_{U,b}f(U,b)
# Proj_{sU}([U,b]-step*gradf-step*etaU*[U,0])
@jit(nogil=True,cache=True)
def computeFactorUpdateSimplex(Ub,step,gradf,sU,eta,bias):
    rk = Ub.shape[1]-bias    
    Ub_new = Ub-step*gradf  
    if (not(sU is None)):
        for j in range(rk):
            Ub_new[:,j] = euclidean_proj_simplex(Ub_new[:,j],sU)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)
    
    if (not(eta is None) and la.norm(Ub_new[:,:rk])>eta):
        Ub_new[:,:rk]=eta*(Ub_new[:,:rk]/la.norm(Ub_new[:,:rk]))
        
    if bias:
        Ub_new[:,-1] = np.clip(Ub_new[:,-1],minb,None)

    Gt = (Ub-Ub_new)
    Gt2 = la.norm(Gt)**2
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

def FUbk(Ubk,Vbk,bk,Xk,lossk,alpha=1,g=1,Vbk_sum=None,bk_sum=None):
    # Vbk,Xk, lossk, alpha are lists 
    # Xv=VvU.T 
    # return f(Ubk) as numpy list and gradF(Ubk)
    
    N=Ubk.shape[0]
    nkv=len(Xk)            
    f=np.array([0.0 for v in range(nkv)]) 
    #if eta is None:
    #    eta=0     
    #if eta>0: f=np.array([0.0 for v in range(nkv+1)]) 

    if (not(isinstance(alpha,list)) and not(isinstance(alpha,np.ndarray))):
        alpha=[alpha for v in range(nkv)]  
    if (not(isinstance(lossk,list)) and not(isinstance(lossk,np.ndarray))):
        lossk=[lossk for v in range(nkv)]  
    
    
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
    
    #if eta>0: f[nkv]=f[nkv]+0.5*eta*la.norm(Ubk)**2 
    f=f.clip(max=maxval)
    if g:
        #gradF=gradF+eta*Ubk
        return f, gradF
    return f