# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import time
import hetroSiCNMFEstimator

model,sources,Xs,rk,etaSweep,outdir,loss=configParser('SiCNMF.config')
if model=='CNMF'
    eta=None
    
V=len(Xs)
assert np.allclose([Xs[v].shape[0] for v in range(V)])
N = [Xs[v].shape[1] for v in range(V)]
Nid={'Patient':Xs[0].shape[0]};Nid.update({sources[v]:N[v] for v in range(V)})


verbose = 1
tol = 1e-6
numIt = 100
gradIt = 20

if __name__ == '__main__':
    jobs=[];
    multProc=1;
    if multProc:
        import multiprocessing
        jobs=[];
        for i in range(5):
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


def run_save(rk,eta,i,verbose):
    np.random.seed()
    print "Model:%s, Rank:%d, eta:%s, i:%d, rand:%f"  %(model,rk,eta,i,np.random.rand())
    fname='%svandy_%s_%s_%d_%d_gradIt_%d' %(outdir,model,eta,rk,i,gradIt)
    t=time.time()
    
    b1s, b2s, Us, f, nit = hetroSiCNMFEstimator.fit(Xs=Xs, N=N, loss=loss, rk=rk, eta=eta, gradIt=gradIt, numIt=numIt, tol=tol, verbose=verbose,fname)
    
    result={'b1s':b1s,'b2s':b2s,'Us':Us,'f':f, 'nit': nit, 't':time.time()-t,'model':model}    
    
    save_file(result,'%s_Factors.pickle' %fname)
    
    
def configParser(fname)
    inputfile = open(fname).readlines()
    outdir="./"
    etaSweep=[1]
    loss = ['sparse_poisson']
    for line in inputfile:
        line=line.replace("'","").replace('"',"")
        (k,v)=line.split(":",1);k=k.strip();v=v.strip()
        if k=='model':
            model=v
        if k=='nFactors':
            rank=int(v)
        if k=='outdir':
            outdir=v
        if k=='sources':
            sources=v.split(,)
        if k=='Xfiles':
            Xs=[np.load(Xf).ravel()[0] for Xf in v.split(,)]
        if k=='etaSweep':
            etaSweep=[float(vv) for vv in v.split(,)]
        if k=='loss':
            loss=[l.strip() for l in v.split(,)]
    if not(len(loss)==len(Xs)):
        loss=[loss[0]]*len(Xs)

    return model,sources,Xs,rank,etaSweep,outdir,loss

