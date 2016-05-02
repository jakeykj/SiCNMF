import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import scipy.linalg as la
import cPickle as pickle
import sicnmf_helper


tol = 1e-6;
verbose = 0;
gradIt = 20;
V=1
rk=2
K=2


FUbk=sicnmf_helper.FUbk
computeFactorUpdate=sicnmf_helper.computeFactorUpdateSimplex

def fit(Xs, N, loss, rk, eta=None, gradIt=10, numIt=50, verbose=0, filename="tmp"):    
    V=len(Xs)
    globals()['verbose'] = verbose
    globals()['gradIt'] = gradIt
    globals()['V'] = V
    globals()['K'] = V+1
    globals()['rk'] = rk
    
    nPat = Xs[0].shape[0]
    n = np.sum([nPat*N[v] for v in range(V)])
   
    for v in range(V):
        assert (Xs[v]).min()>=0,'NonNegative matrices only'    
        assert (sp.issparse(Xs[v]))>=0,'NonNegative matrices only'    
        
    bias=1
    p_bias=0
    # NO PATIENT BIAS
    # Initialization
    if bias: Ubs = [np.hstack((np.random.rand(nPat, rk),p_bias*np.random.rand(nPat,1) ))]+[np.random.rand(N[v], rk+bias) for v in range(V)]    
    else: Ubs = [np.random.rand(nPat, rk)]+[np.random.rand(N[v], rk) for v in range(V)]    
    for v in range(V):
        Ubs[v+1] = normalize(Ubs[v+1],'l1',0)
                
    Xts=[Xs[v].T.tocsc() for v in range(V)]
    Xs=[Xs[v].tocsc() for v in range(V)]
    
    #TODO: smart alpha
    alpha=sicnmf_helper.computeAlpha(Xs,N,loss,rk)

    stat={'f_iter':[],'Codennz':[],'Mednnz':[],'gradIt':[]}
    
    # gradient functions for each factors: input U, V,g
    gfs = [{'func':FUbk, 'args':{'Xk':Xts,'lossk':loss,'alpha':alpha,'eta':eta}}]
    gfs = gfs + [{'func':FUbk, 'args':{'Xk':[Xs[v]],'lossk':[loss[v]],'alpha':[alpha[v]]}} for v in range(V)]
    
    # projected update for each factor for each factor: input U, step, gradU    
    if (eta==None):
        # No sparsity
        nextFactors = [{'func':computeFactorUpdate,'args':{'sU':None,'bias':0}} for k in range(K)]
    else:
        # Sparsity
        nextFactors = [{'func':computeFactorUpdate,'args':{'sU':None,'bias':0}}]
        nextFactors = nextFactors + [{'func':computeFactorUpdate,'args':{'sU':1,'bias':bias}} for v in range(V)]        
      
    # Outer Iterations
    ftol = 1e-3*n    
    ftmp = np.inf
    f=np.array([0.0 for v in range(V+1)])
    nnz=[[]]*V
    

    for i in range(numIt):        
        if verbose>0: print "Iter: %d" %i
        chtol = 1e-6*np.sum([la.norm(Ubs[k])**2 for k in range(K)])
        Ubstmp = [Ubs[k].copy() for k in range(K)]
        git = np.zeros(K)
        change = 0.0 
        
        ###########################################

        for v in range(V):
            k=v+1
            print "Update Factor %d" %k
            if bias:
                Vbk=[np.hstack((Ubs[0][:,:-1],np.ones((nPat,1))))];    
                bk=[Ubs[0][:,-1]]
                Vbk_sum=[np.sum(Vbk[vv],0) for vv in range(len(Vbk))]
                bk_sum=[np.sum(bk[vv]) for vv in range(len(Vbk))]
            else:
                Vbk=[Ubs[0]];    
                bk=[0.0]
                Vbk_sum=[np.sum(Vbk[vv],0) for vv in range(len(Vbk))]
                bk_sum=[0.0 for vv in range(len(Vbk))]                   
            gfs[k]['args'].update({'Vbk':Vbk,'bk':bk,'Vbk_sum':Vbk_sum,'bk_sum':bk_sum})            
            Ubs[k],f[[v,V]],git[k]= singleUbUpdate(Ubs[k],gfs[k],nextFactors[k],gradIt,verbose)
            
            ch=la.norm(Ubs[k]-Ubstmp[k])**2
            change=change+ch
        
        k=0        
        print "Update Patient Factor"
        if bias:
            Vbk=[np.hstack((Ubs[v+1][:,:-1],p_bias*np.ones((N[v],1)))) for v in range(V)];    
            bk=[Ubs[v+1][:,-1] for v in range(V)]
            Vbk_sum=[np.sum(Vbk[v],0) for v in range(len(Vbk))]
            bk_sum=[np.sum(bk[v]) for v in range(len(Vbk))]
        else:
            Vbk=[Ubs[v] for v in range(1,K)];    
            bk=[0.0]
            Vbk_sum=[np.sum(Vbk[v],0) for v in range(len(Vbk))]
            bk_sum=[0.0 for v in range(len(Vbk))]                   
        gfs[k]['args'].update({'Vbk':Vbk,'bk':bk,'Vbk_sum':Vbk_sum,'bk_sum':bk_sum})

        Ubs[k],f[range(V+1)],git[k]=singleUbUpdate(Ubs[k],gfs[k],nextFactors[k],gradIt,verbose)        
        
        ch=la.norm(Ubs[k]-Ubstmp[k])**2
        change=change+ch
        stat['f_iter'].append(f)

        ########################################             
        
        nnz=[[len(np.where(Ubs[v+1][:,j]>=1e-15)[0]) for j in range(rk)] for v in range(V)]
        stat=updateStats(f,nnz,git,stat)
        print("End of Iter:%d. change= %0.4g. fval=%0.4g. Phenotype Sparsity: " %(i, change, f.sum()))
        for v in range(V):
            print("\t Mode %d: median(nnz)=%d/%d, (min_nnz,max_nnz)=(%d,%d)" %(v,np.median(nnz[v]),N[v], np.min(nnz[v]), np.max(nnz[v])))
        
        # Exit condition
        if ((i>=10)  and ((change < chtol)  or (np.abs(ftmp-f.sum())<ftol))):
            exitIter(i,ftmp,f.sum(),change)
            break
        else:
            ftmp=f.sum()
            
    if (i==numIt-1):
        exitIter(i,ftmp,f.sum(),change)        
                  
    stat['niter']=i+1
    return Ubs, f, stat

def updateStats(f,nnz,git,stat):
    # hardcoded numbers
    stat['gradIt'].append(git)
    stat['f_iter'].append(f)
    stat['Codennz'].append(nnz[0])
    stat['Mednnz'].append(nnz[1])
    return stat      
                  
def exitIter(i,fold,fnew,change):
    print "Exited in %d iterations with change (ch=%0.2g); decrease in function (f-fnew<%f); loss=%f" %(i+1,change,fold-fnew,fnew)


def singleUbUpdate(Ub,gf,nextFactor,gradIt,tol):       
    #Ub assumed to be in the domain    
    ftol=1e-2    
    step=1    
    tau=0.1
    ftmp=np.inf
    Ub_curr=Ub.copy()
    Ub_new =Ub.copy()                  
    for i in xrange(gradIt):          
        chtol=1e-8*la.norm(Ub_curr)**2
        
        #Compute Gradients
        f,gradf = gf['func'](Ubk=Ub_curr,g=1,**gf['args'])         
        
        Ub_new,Gt2,m = nextFactor['func'](Ub=Ub_curr,step=step,gradf=gradf,**nextFactor['args'])        
        fnew = gf['func'](Ubk=Ub_new,g=0,**gf['args'])
        if verbose>1: print f
        if verbose>1: print la.norm(Ub_curr-Ub_new),m,(0.5/step)*Gt2,Gt2,m-(0.5/step)*Gt2, f,fnew,np.sum(f-fnew)
        
        # LINE SEARCH
        # Increase step size
        case = -1
        k = 0
        for k in range(50):
            if m-(0.5/step)*Gt2<-1e-10:
                print 'm:',m,'step:',step,'Gt2:',Gt2,'0.5/step*Gt2:',(0.5/step)*Gt2, 'm-0.5/step*Gt2:',m-(0.5/step)*Gt2
            if (np.sum(f-fnew)<=max(m-(0.5/step)*Gt2,0.0)):
                case=1
                break
            
            # Saving latest valid step
            Ubtemp=Ub_new.copy()
            ftemp=fnew.copy()
            mtemp=m
            Gt2temp=Gt2
            
            # Increment
            step=step/tau;
            Ub_new,Gt2,m = nextFactor['func'](Ub=Ub_curr,step=step,gradf=gradf,**nextFactor['args'])
            if verbose>1: print la.norm(Ub_curr-Ub_new),m,(0.5/step)*Gt2,Gt2,m-(0.5/step)*Gt2, f,fnew,np.sum(f-fnew)
            fnew = gf['func'](Ubk=Ub_new,g=0,**gf['args'])
            if (np.sqrt(Gt2)<1e-20):
                case=2
                break
        if k:
            step=step*tau
            Ub_new=Ubtemp.copy()
            fnew=ftemp.copy()
            m=mtemp;
            Gt2=Gt2temp
        else:
            # Decrease step size
            ftemp = np.inf; Ubtemp = Ub_curr.copy(); mtemp=m; Gt2temp=Gt2
            for k in range(k,50):
                if m-(0.5/step)*Gt2<-1e-10:
                    print 'm:',m,'step:',step,'Gt2:',Gt2,'0.5/step*Gt2:',(0.5/step)*Gt2, 'm-0.5/step*Gt2:', m-(0.5/step)*Gt2
                if (np.sum(f)-np.sum(fnew)>=max(m-(0.5/step)*Gt2,0.0)):
                    case=3
                    break
                if (Gt2<1e-50):
                    Ub_new=Ub_curr
                    fnew = f
                    case = 4
                    break 

                step = tau*step
                Ub_new,Gt2,m = nextFactor['func'](Ub=Ub_curr,step=step,gradf=gradf,**nextFactor['args'])
                if verbose>1: print la.norm(Ub_curr-Ub_new),m,(0.5/step)*Gt2,Gt2,m-(0.5/step)*Gt2, f,fnew,np.sum(f-fnew)
                fnew = gf['func'](Ubk=Ub_new,g=0,**gf['args'])

                if ((np.sum(ftemp) < np.sum(fnew)) and ((np.sum(f)-np.sum(ftemp))>max(1e-2*m,0))):
                    step = step/tau
                    Ub_new = Ubtemp.copy()
                    fnew = ftemp.copy()
                    m=mtemp
                    Gt2=Gt2temp
                    case=5
                    break
                else:
                    ftemp = fnew.copy()
                    Ubtemp = Ub_new.copy()
                    mtemp = m
                    Gt2temp = Gt2
                
            if (k>=100):
                print "k>=50,step=%0.2g" %step  
                Ub_new=Ub_curr.copy()
                fnew=f
                Gt2=0

        change = Gt2
        
        if verbose>0:
            print "\t PGDUpdate: change= %0.4g, k=%d, exit_case=%d, step:%0.4g,fnew=%0.4g, f=%0.4g, fdiff=%0.4g>%0.4g: m=%0.4g"\
            %(change,k,case,step,np.sum(fnew),np.sum(f),np.sum(f)-np.sum(fnew),m-(0.5/step)*Gt2,m)        
        
        if change<chtol:
            if verbose:
                print ("Exiting PGD update in %d iterations due to small update change %f" %(i+1,change))
            break
            
        if np.sum(f-fnew)<ftol:
            if verbose:
                print ("Exiting PGD update in %d iterations due to small f change %f" %(i+1,np.sum(f-fnew)))
            break
        
        Ub_curr = Ub_new.copy()
        
            
    if verbose and (i==(gradIt-1)):
        print ("Exiting PGD update in %d iterations" %(i+1))
    return Ub_new,fnew,i+1
