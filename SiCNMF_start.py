# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import time
import hetroSiCNMFEstimator
ids=range(5)
model,sources,Xs,rk,etaSweep,outdir,loss=configParser('SiCNMF.config')
if model=='CNMF'
    eta=None
    
V=len(Xs)
N = [Xs[v].shape[1] for v in range(V)]

verbose = 1
numIt = 100
gradIt = 20

if __name__ == '__main__':
    jobs=[];
    multProc=1;
    if multProc:
        import multiprocessing
        jobs=[];
        for i in range(ids):
            for eta in etaSweep:
                p=multiprocessing.Process(target=run_save,args=(rk,eta,i,verbose))
                jobs.append(p)
                p.start()
        for p in jobs:
            p.join()
    else:
        verbose = 1 
        for eta in etaSweep:     
            np.random.seed(42)
            run_save(rk,eta,0,verbose)

# min \sum_v \alpha_v D_v(X_v, U_pU_v.T) + eta*||U_p||_F^2, s.t. ||U_v[:,j]||_1=1
def run_save(rk,eta,i,verbose):
    np.random.seed()
    print "Model:%s, Rank:%d, eta:%s, i:%d, rand:%f \n"  %(model,rk,eta,i,np.random.rand()),Xs
    fname='%svandy_%s_%s_eta%0.2g_i%d_rk%d' %(outdir,model,eta,i,rk)
    t=time.time()
    
    b1s, b2s, Us, f, nit = hetroSiCNMFEstimator.fit(Xs=Xs, Nid=Nid, loss=loss, rk=rk, eta=eta, gradIt=gradIt, numIt=numIt, verbose=verbose,fname)
    
    result={'b1s':b1s,'b2s':b2s,'Us':Us,'f':f, 'nit': nit, 't':time.time()-t,'model':model}    
    
    save_file(result,'%s_Factors.pickle' %fname)
    
    
def configParser(fname):
    inputfile = open(fname).readlines()
    outdir="./"
    etaSweep=[1]
    loss = ['sparse_poisson']
    for line in inputfile:
        line=line.replace("'","").replace('"',"")
        (k,v)=line.split(":",1);k=k.strip();v=v.strip()
        if k=='model':
            # CNMF/SiCMNF
            model=v
        if k=='nFactors':
            # Rank
            rank=int(v)
        if k=='outdir':
            # Output directory (end with /)
            outdir=v
        if k=='sources':
            # Souces of data, medication, diagnosis etc
            sources=v.split(',')
        if k=='Xfiles':
            # Data file paths
            Xs=[np.load(Xf).ravel()[0] for Xf in v.split(',')]
        if k=='etaSweep':
            #  eta parameters
            etaSweep=[float(vv) for vv in v.split(',')]
        if k=='loss':
            loss=[l.strip() for l in v.split(',')]
        if k=='ids':
            ids=[float(vv) for vv in v.split(',')]
    if not(len(loss)==len(Xs)):
        loss=[loss[0]]*len(Xs)

    return model,sources,Xs,rank,etaSweep,outdir,loss