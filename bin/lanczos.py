import numpy as np
import scipy as sp
from IPython.display import clear_output

def lanczos_reorth(A,v,k,reorth=0):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    v : starting vector
    k : number of iterations (matvecs)
    reorth : number of iterations to reorthogonalize
    
    Output
    ------
    Q : Lanczos vectors
    α : diagonal coefficients
    β : off diagonal coefficients 
    """
    
    n = A.shape[0]
    
    Q = np.zeros((n,k+1),dtype=A.dtype)
    α = np.zeros(k,dtype=A.dtype)
    β = np.zeros(k,dtype=A.dtype)
    
    Q[:,0] = v / np.linalg.norm(v)
    
    for i in range(k):

        qip1 = A@Q[:,i] - β[i-1]*Q[:,i-1] if i>0 else A@Q[:,i]
        
        α[i] = Q[:,i].conj().T@qip1
        qip1 -= α[i]*Q[:,i]
        
        if reorth>i:
            qip1 -= Q[:,:i-1]@(Q[:,:i-1].conj().T@qip1)
            
        β[i] = np.linalg.norm(qip1)
        Q[:,i+1] = qip1 / β[i]
                
    return Q,(α,β)


def lanczos_FA(f,Q,α,β,k,normb=1):
    """
    compute Lanczos-FA iterate
    
    Input
    -----
    k : degree of approximation
    
    """

    if k==0:
        return np.zeros_like(Q[:,0])
    
    Θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1],tol=1e-30)
    
    return normb*(Q[:,:k]@(S@(f(Θ)*(S.T[:,0]))))
