# -*- coding: utf-8 -*-
import numpy as np
import cPickle as pickle
import time
import hetroSiCNMFEstimator_noalpha as hetroSiCNMFEstimator
import sys,getopt
import seaborn as sns
    


# min \sum_v \alpha_v D_v(X_v, U_pU_v.T) + eta*||U_p||_F^2, s.t. ||U_v[:,j]||_1=1
def run_save(rk,eta,i,verbose):
    np.random.seed()
    print i,eta
    if i<=20: sys.stdout = open(outdir+time.strftime("%d%m")+ '_%s_eta%s_i%d_noalpha' %(model,eta,i) + ".out", "w")
    print "ALL PARALLEL PROCESS SAME RANDOM SEED"
    print "Model:%s, Rank:%d, eta:%s, i:%d, rand:%f \n"  %(model,rk,eta,i,np.random.rand()),Xs
    fname='%svandy_%s_eta%s_i%d_rk%d' %(outdir,model,eta,i,rk)
    print fname
    t=time.time()

    Ubs, f, stat = hetroSiCNMFEstimator.fit(Xs=Xs, N=N, loss=loss, rk=rk, eta=eta, gradIt=gradIt, numIt=numIt, verbose=verbose,filename=fname)
    print "Done Fitting"
    
    result={'Ubs':Ubs,'f':f, 'stat': stat, 't':time.time()-t,'model':model}        
    pickle.dump(result,open('%s.pickle' %fname,'wb'))
    print "Saved Results"
    
    sns.plt.switch_backend('agg')
    sns.plt.subplot(1, 3, 1); sns.plt.plot([np.sum(stat['f_iter'][k]) for k in range(len(stat['f_iter']))])
    sns.plt.subplot(1, 3, 2); sns.plt.plot([np.median(stat['Codennz'][k]) for k in range(len(stat['Codennz']))])
    sns.plt.subplot(1, 3, 3); sns.plt.plot([np.median(stat['Mednnz'][k]) for k in range(len(stat['Mednnz']))])
    sns.plt.tight_layout()    
    
    if not(fname==None):
        sns.plt.savefig('%s_stats.png' %fname)
    else:
        filename = 'stats_last_run.png'
        sns.plt.savefig(filename)
    print "Saved Stats"
    
    
def configParser(fname):
    inputfile = open(fname).readlines()
    outdir="./"
    etaSweep=[1]
    loss = ['sparse_poisson']
    for line in inputfile:
        if line.startswith("#"):
            continue
        line=line.replace("'","").replace('"',"")        
        if not(len(line)):
            continue
        (k,v)=line.split(":",1);
        k=k.strip();
        v=v.strip()
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
            etaSweep=[vv for vv in v.split(',')]
            for ii in range(len(etaSweep)):
                if (etaSweep[ii]=='None'):
                    etaSweep[ii]=None
                else:
                    etaSweep[ii]=float(etaSweep[ii])
        if k=='loss':
            loss=[l.strip() for l in v.split(',')]
        #if k=='ids':
        #    ids=[float(vv) for vv in v.split(',')]
    if not(len(loss)==len(Xs)):
        loss=[loss[0]]*len(Xs)

    return model,sources,Xs,rank,etaSweep,outdir,loss

if __name__ == '__main__':
   
    ids = 0
    cfg_file = 'SiCNMF.config'
    multProc=1;
    try:
        opts,args=getopt.getopt(sys.argv[1:],"hf:i:p:",["config-file=","run_ids=","parallel="])
    except getopt.GetoptError:
        print('SiCNMF_start.py -f <config file> -i <run_id>')
        
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print('SiCNMF_start.py -f <config file> -i <run_id>')
            sys.exit(0)
        elif opt in ('-f','--config-file'):
            # Config file
            cfg_file=str(arg)          
        elif opt in ('-i','--run_ids'):
            # Comma separated run_ids (multiple initializations). Default 1
            ids=[int(a) for a in arg.split(',')]
        elif opt in ('-p','--parallel'):
            # number of cores for multiprocessing code, 0 for no parallel
            multProc=int(arg)
    
    
    model,sources,Xs,rk,etaSweep,outdir,loss=configParser(cfg_file)
    V=len(Xs)
    N = [Xs[v].shape[1] for v in range(V)]

    verbose = 1
    numIt = 50
    gradIt = 50
    print ids
    print etaSweep
    jobs=[];   
    if multProc:
        import multiprocessing
        pool = multiprocessing.Pool(multProc)
        jobs=[];
        for eta in etaSweep:
            for i in ids:     
                jobs.append((rk,eta,i,verbose))
        print jobs
        results=[pool.apply_async(run_save,p) for p in jobs]
        for result in results:
            result.get()
        
    else:
        verbose = 1
        eta=0.5
        numIt=2
        print "debug mode:multiproc =0"
        np.random.seed(42)
        run_save(rk,eta,ids[0],verbose)
