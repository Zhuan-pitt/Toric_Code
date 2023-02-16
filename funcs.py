import numpy as np
from matplotlib import pyplot
import copy
from scipy import linalg


def row_contract44(T1,T2):
    #T1,T2 4th order tensor
    # (2)|   |   
    #(0)-T1--T2-(1)
    # (3)|   | 
    T = np.tensordot(T1,T2,([1],[0]))
    s = np.shape(T)
    T = np.transpose(T,[0,3,1,4,2,5])
    T = np.reshape(T,[s[0],s[3],s[1]*s[4],s[2]*s[5]])
    return T


def row_contract32(T1,T2):          
    #(0)-T1--T2-(2)
    #    |(1)    
    T = np.tensordot(T1,T2,([1],[0]))
    T = np.transpose(T,([0,2,1]))
    return T
def row_contract23(T1,T2):          
    #(0)-T1--T2-(1)
    #        |(2)    
    T = np.tensordot(T1,T2,([1],[0]))
    return T



def col_contract33(T1,T2):
    #T1,T2 3th order tensor   
    #(0)--T1--(1)
    #     |        -->  (0)--T--(1)
    #(2)--T2--(3)  
    T = np.tensordot(T1,T2.conj(),([2],[2]))
    s = np.shape(T)
    T = np.transpose(T,[0,2,1,3])
    T = np.reshape(T,[s[0]*s[2],s[1]*s[3]])
    return T

def col_contract34(T1,T2):  
    #(0)--T1--(1)
    #     |        -->  --T--
    #(2)--T2--(3)         |
    #     |(4)  
    T = np.tensordot(T1,T2,([2],[2]))
    s = np.shape(T)
    T = np.transpose(T,[0,2,1,3,4])
    T = np.reshape(T,[s[0]*s[2],s[1]*s[3],s[4]])
    return T


def col_contract43(T1,T2):  
    #  (2)|  
    #(0)--T1--(1)         |
    #     |        -->  --T--
    #(3)--T2--(4)            

    T = np.tensordot(T1,T2.conj(),([3],[2]))
    s = np.shape(T)
    T = np.transpose(T,[0,3,1,4,2])
    T = np.reshape(T,[s[0]*s[3],s[1]*s[4],s[2]])
    return T

def col_contract343(T1,T2,T3):
    #T1,T2 3th order tensor   
    #--T1--
    #  |        
    #--T2--   -->  --T--
    #  | 
    #--T3--
    T = col_contract34(T1,T2)
    T = col_contract33(T,T3)
    return T


def sqrt_left_right(Ml,Mr,threshold=1e-14):
    #XX^dag = Mr
    #YY^dag = Ml
    lamr,vr = np.linalg.eigh(Mr)

    idxr = lamr.argsort()[::-1]
    if lamr[idxr[0]]<threshold and abs(lamr[idxr[-1]])>threshold:
        lamr,vr = np.linalg.eigh(-Mr)
        idxr = lamr.argsort()[::-1]

    
    laml,vl = np.linalg.eigh(Ml)
    idxl = laml.argsort()[::-1]
    if laml[idxl[0]]<threshold and abs(laml[idxl[-1]])>threshold:
        laml,vl = np.linalg.eigh(-Ml)
        idxl = laml.argsort()[::-1]

    tot = lamr[idxr]*laml[idxl]

    #print(lamr[idxr])
    #print(laml[idxl])
    
    num = np.sum(tot>threshold**2)
    assert num >0, print(lamr[idxr],lamr[idxl])
    sqrtlamr = np.sqrt(lamr[idxr[0:num]])
    X = vr[:,idxr[:num]]@np.diag(sqrtlamr)
    
    sqrtlaml = np.sqrt(laml[idxl[0:num]])
    Y = vl[:,idxl[:num]]@np.diag(sqrtlaml)
    assert num>0, 'positive eigenvalue expected '+f"lam = {lamr} "
    return Y,X

def sqrthm(A,threshold=1e-10):
    #sqrt of the hermitian matrix A
    #discard the eigenvalues that smaller than threshold
    #XX^dag = A

    lam,vr = np.linalg.eigh(A)
    idx = lam.argsort()[::-1]
    if lam[idx[0]]<threshold and abs(lam[idx[-1]])>threshold:
        lam,vr = np.linalg.eigh(-A)
        idx = lam.argsort()[::-1]

    
    num = np.sum(lam[idx]>threshold)
    sqrtlam = np.sqrt(lam[idx[0:num]])
    X = vr[:,idx[:num]]@np.diag(sqrtlam)
    assert num>0, 'positive eigenvalue expected '+f"lam = {lam} "
    return X



def delta_tensor(N,m):
    #return the N-th order delta tensor
    T = np.zeros([m]*N)
    for i in range(m):
        
        T[tuple([i]*N)] = 1
    return T

def single_trans(h,u1="I",u2="I"):
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=np.exp(h/2),np.exp(h/2),np.exp(-h/2),np.exp(-h/2)
    
    
    matrix_list={"I":np.eye(2),"X":np.matrix([[0,1],[1,0]])} 
    u1 = matrix_list[u1]
    u2 = matrix_list[u2]
    
    A2 = np.tensordot(A,u1,([3],[0]))

    A3 = np.tensordot(np.tensordot(P,A2,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    A3 = np.reshape(A3,[2,2,2,2,4])
    
 
    A2s = np.tensordot(A,u2,([3],[0]))

    A3s = np.tensordot(np.tensordot(P,A2s,([1],[0])),P,([4],[0]))
    A3s = np.transpose(A3s,[0,2,3,4,1,5])
    A3s = np.reshape(A3s,[2,2,2,2,4])
    
    dA3 = np.tensordot(A3,A3s,([4],[4]))
    dA3 = np.transpose(dA3,[0,4,1,5,2,6,3,7])
    dA3 = np.reshape(dA3,[4,4,4,4])
    return dA3