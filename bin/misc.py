import numpy as np

def model_problem_spectrum(n,ρ,κ=1e3,dtype='float'):
    '''
    returns spectrum of model problem
    '''
    λ1 = 1
    λn = κ
    Λ = λ1+(λn-λ1)*np.arange(n)/(n-1)*ρ**np.arange(n-1,-1,-1,dtype=dtype)
    
    return Λ


def get_discrete_nodes(L,k,X=None):
    
    Xn = np.zeros(k,dtype=np.longdouble)
    Xn[:] = np.nan
    
    i0 = 0
    if X is not None:
        Xn[:len(X)] = X
        i0 = len(X)
    
    for i in range(i0,k):
        for l in L[::-1,0]:
            if np.all(Xn != l):
                Xn[i] = l
                break
    
    return np.sort(Xn)

def get_cheb_nodes(a,b,k):
    """
    Get k Chebyshev T nodes on [a,b]
    """
    nodes = (a+b)/2 + (b-a)*np.cos((2*np.arange(k,0,-1)-1)/(2*k)*np.pi)/2
    return nodes.astype(np.longdouble)
