import numpy as np
import scipy.sparse as sp
import  scipy.linalg as la
from sklearn.preprocessing import normalize

maxval = 1e30
eps = np.sqrt(np.finfo(np.float).eps)
minb=0


def Fcol(w,A,x,b=0.0,alpha=1.0,loss='sparse_poisson'):
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

def FUbkdense(Ubk,Vbk,Xk,lossk,alpha=1,g=1):
    # Vbk,Xk, lossk, alpha are lists 
    # return f(Ubk) as list and gradF(Ubk)
    nkv=len(Xk)
    r=Ubk.shape[1]    
        
    fest=np.array([0.0 for v in range(nkv)])
        
    if g: gradF=np.zeros(Ubk.shape)
    
    if (not(isinstance(alpha,list))):
        alpha=[alpha for v in range(nkv)]

    for j in range(Ubk.shape[0]):        
        for v in range(nkv):
            f = F(w=Ubk[j,:],A=Vbk[v],x=Xk[v].getcol(j),b=0,loss=lossk[v],g=g)
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

# Proj gradient update for min_U,b f(U,b)+etaU ||U||_F^2 s.t. U^(r)\in sU*\Delta_1: gradf=\Nabla_{U,b}f(U,b)
# Proj_{sU}([U,b]-step*gradf-step*etaU*[U,0])
def computeFactorUpdateSiCNMF(Ub,step,gradf,sU,etaU,rk):
    Ub_new = Ub-step*gradf
    if (not(etaU==None)):
        Ub_new[:,:rk] = Ub_new[:,:rk]-step*etaU*Ub[:,:rk]
    if (not(sU==None)):
        Ub_new[:,:rk] = np.apply_along_axis(euclidean_proj_simplex,0,Ub_new[:,:rk],sU)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    else:
        Ub_new[:,:rk] = np.clip(Ub_new[:,:rk],0.0,None)
        Ub_new[:,rk:] = np.clip(Ub_new[:,rk:],minb,None)
    
    Gt = (Ub-Ub_new)
    Gt2 = np.sum(Gt*Gt)
    m = np.sum(gradf*Gt)
    return Ub_new,Gt2,m

def soft_thres(v,s=1):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    u=v.copy()
    u[u<0]=0.0
    return np.clip(u-s,0.0,None)

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
    min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
    n-dimensional vector to project
    s: int, optional, default: 1,
    radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
    Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
    International Conference on Machine Learning (ICML 2008)
    http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if np.abs(v.sum()-s)<1e-15 and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    
    #floating point error fixes
    if (np.abs(v.clip(min=0).sum()-s)<1e-15):
        w = v.clip(min=0)
        w = s*w/w.sum()
        return w
    
    # get the array of cumulative sums of a sorted (decreasing) copy of v       
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)    
    # get the number of > 0 components of the optimal solution    
    try:
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    except IndexError:
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        #print "Warning: IndexError in projection, s=%f" %s
        return w
    if (rho==0):
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        return w;        
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    #compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    if (w.sum()==0):
        w=np.zeros(n);i=np.argmax(v);w[i]=s
        #print "Warning: w was set to zero, s=%f" %s
        return w
    
    #w=(s/np.sum(w))*w  
    assert np.alltrue(w>=0), np.where(w<0)
    
    return w 


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

def euclidean_proj_simplex_k(v, k=None):   
    '''Tasos's method'''
    n, = v.shape # will raise ValueError if v is not 1-D
    if (k==None):
        k=n
    vid=np.argsort(v)[::-1][:k];
    w=np.zeros(v.shape)
    tau=(1.0/k)*(np.sum(v[vid])-1)
    w[vid]=(v[vid]-tau).clip(min=0)
    return w 

def euclidean_proj_nnl1ball(v, s=1):
    '''Uses Result from Tandon, Sra paper'''
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of non-negative values
    u = v.copy(); 
    u[u<0]=0;
    # check if u is already a solution
    if u.sum() <= s:
        return u    
    
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    return euclidean_proj_simplex(u, s=s)

