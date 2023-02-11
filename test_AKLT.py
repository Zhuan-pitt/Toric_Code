import numpy as np
from matplotlib import pyplot
from scipy import linalg
import copy
import iMPS
from scipy.sparse import linalg



def AKLT_ini(L):
    A = np.zeros([2,2,3])
    #transfer from spin1 to 2 spin 1/2 triplet
    A[0,0,0] = 1 
    A[1,1,2] = 1
    A[1,0,1], A[0,1,1] = np.sqrt(1/2),np.sqrt(1/2)

    C = np.zeros([2,2])
    #project to singlet
    C[1,0],C[0,1] = np.sqrt(1/2),-np.sqrt(1/2)

    T  = np.tensordot(A,C,([1],[0]))
    T = np.transpose(T,[0,2,1])

    AKLT_chain = iMPS.iMPS()
    AKLT_chain.construct_from_tensor_list([T]*L)
    return AKLT_chain

def check_eig(AKLT_chain,site):

    trans = AKLT_chain.transfer_matrix(site)
    lam = linalg.eigs(trans,2)[0]
    lam = np.sort(lam)
    assert np.linalg.norm(lam - [-0.25,0.75]) <= 1e-12



if __name__ == "__main__":
    L=10
    AKLT_chain = AKLT_ini(L)
    for i in range(L):
        check_eig(AKLT_chain,i)

    print("Everything passed")