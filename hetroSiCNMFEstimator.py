import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import scipy.linalg as la
import cPickle as pickle
import utils

verbose = 0;
tol = 1e-6;
debug = 1;
gradIt = 20;
V=1
rk=2
K=2

FUbkdense=utils.FUbkdense
computeFactorUpdateSiCNM=utils.computeFactorUpdateSiCNMF

def fit(Xs, N, loss, rk, eta=None, gradIt=10, numIt=50, tol=1e-6, verbose=0, filename="tmp"):    
    V=len(Xs)
    globals()['verbose'] = verbose
    globals()['tol'] = tol 
    globals()['gradIt'] = gradIt
    globals()['V'] = V
    globals()['K'] = V+1
    globals()['rk'] = rk
    
    nPat=Xs[0].shpae[0]
    n=np.sum([nPat*N[v] for v in range(V)])
   
    for v in range(V):
        assert (Xs[v]).min()>=0,'NonNegative matrices only'    
        
    # Initialization
    Ubs = [np.random.rand(nPat, rk+V)]+[np.hstack((np.random.rand(N[v], rk), np.zeros((N[v],V)))) for v in range(V)]    
    for v in range(V):
        Ubs[v+1][:,rk+v] = np.random.rand(N[v],)
        Ubs[v+1] = normalize(Ubs[v+1],'l1',1)
                
    Xts=[Xs[v].T.tocsc() for v in range(V)]
    Xs=[Xs[v].tocsc() for v in range(V)]
    #TODO: smart alpha
    alpha=[1]*V

    stat={'fiter':[],'Codennz':[],'Mednnz':[],'nzCol':[]}
    
    gfs = {'func':FUbkdense, 'args':{'Xk':Xts,'lossk':loss,'alpha':alpha}}
    gfs = gfs + [{'func':FUbkdense, 'args':{'Xk':[Xs[v]],'lossk':[loss[v]],'alpha':[alpha[v]]}} for v in range(V)]
    if (eta==None):
        nextFactors = [{'func':computeFactorUpdateSiCNMF,'args':{'sU':None,'etaU':None,'rk':rk}} for k in range(K)]
    else:
        nextFactors = {'func':computeFactorUpdateSiCNMF,'args':{'sU':None,'etaU':eta,'rk':rk}}
        nextFactors = nextFactors + [{'func':computeFactorUpdateSiCNMF,'args':{'sU':1,'etaU':None,'rk':rk}} for v in range(V)]        
      
    # Outer Iterations
    ftmp=np.inf
    for i in range(numIt):
        if verbose>0: print "Iter: %d" %i
        Ubs,f,change,stat = updateCNMF(Ubs,gfs,nextFactors,stat)        
        # Exit condition
        if (change < tol or (i>=9 and np.abs(ftmp-f.sum())<1e-3*n)):
            exitIter(i,ftmp,f.sum(),change)
            break
        else:
            ftmp=f.sum()
            
    if (i==numIt-1):
        exitIter(i,ftmp,f.sum(),change)
    
    # Normalization
    b1s, b2s, Us = normalizeFactors(Ubs)       
    
    return b1s, b2s, Us, f[:V], i+1


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


def singleUbUpdate(Ub,gf,nextFactor,gradIt,tol):
    
    
    
    return Ub,f,r,i


def normalizeFactors(Ubs)
    Us=[Ubs[k][:,:rk] for k in range(K)]
    b1s=[Ubs[0][:,rk+v] for v in range(V)]
    b2s=[Ubs[v+1][:,rk] for v in range(V)]    
    return b1s,b2s,Us

def exitIter(i,fold,fnew,change)
    print "Exited in %d iterations with change (ch=%0.2g); decrease in function (f-fnew<%f); loss=%f" %(i+1,change,fold-fnew,fnew)