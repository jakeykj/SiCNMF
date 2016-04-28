import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from sklearn.preprocessing import normalize

maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=0


def F_collective(w,A,x,b=0.0,alpha=1.0,loss='sparse_poisson'):
    V=len(A)
    if not(isinstance(loss,list)):
        loss=[loss for v in range(V)]
    if not(isinstance(b,list)):
        b=[b for v in range(V)]
    if not(isinstance(alpha,list)):
        alpha=[alpha for v in range(V)]

    f=np.array([0.0 for v in range(V)])
    gradF=np.zeros(w.shape)
    for v in range(V):        
        fv,gradfv=F(w,A[v],x[v],b[v],loss[v])
        try: 
            f[v]=alpha[v]*fv         
            gradF=gradF+alpha[v]*gradfv 
        except OverflowError:
            print f.dtype,fv,alpha[v],alpha[v]*fv
            raise OverflowError
    f=f.clip(max=maxval)
    return f,gradF



def FUbk(Ubk,Vbk,Xk,lossk,Sk=None,alpha=1,g=1):
    # Vbk,Xk, lossk, alpha are lists 
    # return f(Ubk) as list and gradF(Ubk)
    nkv=len(Xk)
    r=Ubk.shape[1]    
    if Sk==None:
        Sk=[np.ones(r,) for v in range(nkv)]
    elif not(isinstance(Sk,list)):
        Sk=[Sk for v in range(nkv)]
        
    fest=np.array([0.0 for v in range(nkv)])
        
    if g: gradF=np.zeros(Ubk.shape)
    
    if (not(isinstance(alpha,list))):
        alpha=[alpha for v in range(nkv)]

    for j in range(Ubk.shape[0]):        
        for v in range(nkv):
            f = F(w=Ubk[j,:],A=Vbk[v]*Sk[v],x=Xk[v].getcol(j),b=0,loss=lossk[v],g=g)
            if g:
                fest[v]=fest[v]+alpha[v]*f[0]
                gradF[j,:]=gradF[j,:]+alpha[v]*f[1]
            else:
                fest[v]=fest[v]+alpha[v]*f
                  
    fest=fest.clip(max=maxval)
    if g: return fest, gradF
    else: return fest

    
def computeAlpha(Xs,entV,N,loss,rk):
    return np.array([1 for v in range(len(Xs))])

# If trunc=False, project the updated factors to sU simplex, else project onto k sparse simplex 
def computeFactorUpdateSimplex(Ub,step,gradf,sU,eta,bias):
    Ub_new = Ub-step*gradf
    if (not(sU==None)):
        Ub_new[:,:rk] = np.apply_along_axis(euclidean_proj_simplex,0,Ub_new[:,:rk],sU)
        Ub_new[:,rk:] = np.clip(Ub_new[:,-1],minb,None)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

####################################################

def F(w,A,x,b=0.0,loss='sparse_poisson',g=1):    
    f=0.0
    if g: gradF=np.zeros(w.shape)
    
    if sp.issparse(x):
        if (loss.endswith('gaussian')):
            xhat= A.dot(w)
            if b: xhat += b
            f = 0.5*(la.norm(xhat[x.indices]-x.data,2)**2)+0.5*(la.norm(xhat)**2-la.norm(xhat[x.indices])**2)
            if g: gradF = (A.T).dot(xhat)-((A[x.indices]).T).dot(x.data)
        elif (loss.endswith('poisson')):
            xhat_ind=(A[x.indices,:]).dot(w)
            if b: xhat_ind += b
            if (np.any(xhat_ind==0)): 
                f = maxval
                if g:
                    weps = w.copy()
                    for i in range(len(w)):
                        weps[i] += eps
                        xhateps_ind = A[x.indices].dot(weps)+b 
                        if np.any(xhateps_ind==0):
                            feps = maxval
                        else:
                            feps= np.sum(A,0).dot(weps)-np.sum(x.data)+np.sum(x.data*np.log(x.data/xhateps_ind))
                        gradF[i] = (feps-f)/eps
                        weps[i] -= eps
            else:
                t = (x.data/xhat_ind) #np.clip((x.data/xhat[x.indices]),None,1e50)
                a = np.sum(A,0)
                f = a.dot(w)-np.sum(x.data)+np.sum(x.data*np.log(t))
                if g: gradF = a-A[x.indices,:].T.dot(t)
            f=min(f,maxval)
        if g: return f,gradF 
        else: return f
    else: 
        print "F is defined for sparse x only" 

####################################

def computeFactorUpdateKsparse(Ub,step,gradf,sU):
    if (la.norm(step*gradf)<1e-30):
        Ub_new=Ub
    else:
        Ub_new = Ub-step*gradf
        if (not(sU==None)):
            Ub_new[:,:-1] = np.apply_along_axis(euclidean_proj_simplex_k,0,Ub_new[:,:-1],sU)
        else:
            Ub_new[:,:-1] = np.clip(Ub_new[:,:-1],0.0,None)
            
        Ub_new[:,-1] = np.clip(Ub_new[:,-1],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

def computeFactorUpdateNNl1(Ub,step,gradf,sU,rk):
    Ub_new = Ub-step*gradf
    if (not(sU==None)):
        Ub_new[:,:rk] = np.apply_along_axis(euclidean_proj_nnl1ball,0,Ub_new[:,:rk],sU)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)           
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

def computeFactorUpdateSoftThreshold(Ub,step,gradf,sU,rk):
    Ub_new = Ub-step*gradf
    if (not(sU==None)):
        Ub_new[:,:rk] = np.apply_along_axis(soft_thres,0,Ub_new[:,:rk],step*sU)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk]-step*Ub[:,:rk],0.0,None)           
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m


def computeFactorUpdateSoftThreshold(Ub,step,gradf,rk,sU=1):
    Ub_new = Ub-step*gradf
    if (not(sU==None)):
        Ub_new[:,:rk] = np.apply_along_axis(soft_thres,0,Ub_new[:,:rk],step*sU)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk]-step*Ub[:,:rk],0.0,None)           
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m



def updateCNMF(Ubs,gfs,nectFactors,stat):
    change=0;
    f=np.array([0.0 for v in range(V)])
    s = np.ones(rk+V,)
    #update pid
    k=0
    Ubtemp=Ubs[k].copy()
    Vbk=[Ubs[v] for v in range(1,K)];    
    gfs[k]['args']['Vbk']=Vbk    
    Ubs[k],f[:V],f[V+k],git=singleUbUpdate(Ubs[k],gfs[k],nextFactors[k],gradIt,tol,verbose)
    
    ch=la.norm(Ubs[k]-Ubtemp)
    change=change+ch**2
    
    if (verbose>0):
        print "UpdateUbk(%d):ch=%0.2g,loss=%f,gradIt=%d"\
        %(Ubs[k].shape[0],ch,f.sum(),git)
        print "nz/nnz=%d/%d (minnz,maxnz)=(%d,%d)" \
        %(np.median([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]), Ubs[pid].shape[0],
          np.min([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]), 
          np.max([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]))
            
           
    #update other factors
    for v in range(V):
        k=v+1
        Sk=np.array([int(j<rk or j==rk+v) for j in range(rk+V)])
        Ubtemp=Ubs[k][:,Sk>0].copy()
        Vbk=[Ubs[0][:,Sk>0]];
        gfs[k]['args']['Vbk']=Vbk        
        Ubs[k][:,Sk>0],f[v],f[V+k],git= singleUbUpdate(Ubs[k][:,Sk>0],gfs[k],nextFactors[k],gradIt,tol,verbose)

        ch=la.norm(Ubs[k][:,Sk>0]-Ubtemp)
        change=change+ch**2

        if (verbose>0):
            print "UpdateUbk(%d):ch=%0.2g,loss=%f,gradIt=%d"\
            %(Ubs[k].shape[0],ch,f.sum(),git)
            print "nz/nnz=%d/%d (minnz,maxnz)=(%d,%d)" \
            %(np.median([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]), Ubs[k].shape[0],
              np.min([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]), 
              np.max([len(np.where(Ubs[k][:,j]<1e-15)[0]) for j in range(rk)]))
                
    change=np.sqrt(change)
        
    return Ubs,f,change
