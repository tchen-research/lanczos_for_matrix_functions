import numpy as np
import scipy as sp
import warnings
from .barycentric import compute_barycentric_weights, barycentric

def remez(f,Λ,k,i=None,tol=1e-10,max_iter=500):
    
    n = len(Λ)
    E = np.inf
    if i is None:
        i = list(range(0,n,n//(k+2)))[:k+2]
    X = Λ[i]
    
    for j in range(max_iter):
        
        σ = (-1)**np.arange(k+2)

        # find equiosillating polynomial pn on X
        w = compute_barycentric_weights(X)

        # compute osillation size
        h = (w@f(X))/(w@σ)

        # compute values of p_X so that it equiosillatse f with distance h at points in X
        p_X = f(X) - h * σ

        # get function for p
        p = lambda x: barycentric(x,X,p_X,w)

        err_Λ = f(Λ)-p(Λ)
        err_X = f(X)-p(X)

        if np.max(np.abs(err_Λ))-np.abs(h)<tol:
            return i,p,np.max(np.abs(err_Λ)),np.abs(h)

        if j==max_iter-1:
            warnings.warn(f"failed to converge to specified tollerence for degree {k}")
            return i,p,np.max(np.abs(err_Λ)),np.abs(h)

        # update reference
#        zeros = []
#        for j in range(n-1):
#            if np.sign(err_Λ[j])!=np.sign(err_Λ[j+1]):
#                zeros.append(j)
        zeros = np.nonzero(np.diff(np.sign(err_Λ)))[0]


        nz = len(zeros)
        i_new = np.zeros(nz+1,dtype='int')

        rb = zeros[0]
        i_new[0] = np.argmax(np.sign(err_Λ[rb])*err_Λ[:rb+1])
        for j in range(1,nz):
            lb = zeros[j-1]+1
            rb = zeros[j] 
            if lb==rb:
                i_new[j] = lb
            else:
                i_new[j] = lb+np.argmax(np.sign(err_Λ[rb])*err_Λ[lb:rb])

        lb = zeros[nz-1]
        i_new[nz] = lb + np.argmax(np.sign(err_Λ[-1])*err_Λ[lb:])

        g_max = np.argmax(np.abs(f(Λ[i_new]) - p(Λ[i_new])))
        d = np.max([g_max-(k+1),0])

        i = i_new[d:d+k+2]
        X = Λ[i]

def remez_zero(f,Λ,k,i=None,tol=1e-10,max_iter=500):
    
    n = len(Λ)
    E = np.inf
    if i is None:
        i = list(range(0,n,n//(k+2)))[:k+2]
    X = Λ[i]
    
    for j in range(max_iter):
        
        σ = (-1)**np.arange(k+2)

        # find equiosillating polynomial pn on X
        w = compute_barycentric_weights(X)

        # compute osillation size
        h = 1

        # compute values of p_X so that it equiosillatse zero at points in X
        p_X = h * σ

        # get function for p
        
        p = lambda x: barycentric(x,X,p_X,w)/barycentric(0,X,p_X,w)

        err_Λ = p(Λ)
        err_X = p(X)

        if np.max(np.abs(err_Λ))-1/barycentric(0,X,p_X,w)<tol:
            return i,p,np.max(np.abs(err_Λ)),1/barycentric(0,X,p_X,w)

        if j==max_iter-1:
            warnings.warn(f"failed to converge to specified tollerence for degree {k}")
            return i,p,np.max(np.abs(err_Λ)),1/barycentric(0,X,p_X,w)


        zeros = np.nonzero(np.diff(np.sign(err_Λ)))[0]

        nz = len(zeros)
        i_new = np.zeros(nz+1,dtype='int')

        rb = zeros[0]
        i_new[0] = np.argmax(np.sign(err_Λ[rb])*err_Λ[:rb+1])
        for j in range(1,nz):
            lb = zeros[j-1]+1
            rb = zeros[j] 
            if lb==rb:
                i_new[j] = lb
            else:
                i_new[j] = lb+np.argmax(np.sign(err_Λ[rb])*err_Λ[lb:rb])

        lb = zeros[nz-1]
        i_new[nz] = lb + np.argmax(np.sign(err_Λ[-1])*err_Λ[lb:])

        g_max = np.argmax(np.abs(p(Λ[i_new])))
        d = np.max([g_max-(k+1),0])

        i = i_new[d:d+k+2]
        X = Λ[i]
