import numpy as np
from matplotlib import pyplot
import copy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator

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

def row_contract33(T1,T2):
    #(0)-T1--T2-(1)
    # (2)|   |(3)    
    T = np.tensordot(T1,T2,([1],[0]))
    T = np.transpose(T,[0,2,1,3])
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


def col_contract343_sparse(T1,T2,T3):
    #T1,T2 3th order tensor   
    #--T1--
    #  |        
    #--T2--   -->  --T-- (linear operator)
    #  | 
    #--T3--
    bra = T1
    ope = T2
    ket = T3
    
    s1 = np.shape(bra) 
    s2 = np.shape(ope) 
    s3 = np.shape(ket) 
    def mv(v):
                V = np.reshape(v,[s1[1],s2[1],s3[1]])
                M = np.tensordot(bra,V,([1],[0]))
                M = np.tensordot(M,ope,([2,1],[1,2]))
                M = np.tensordot(M,ket.conj(),([3,1],[2,1]))
                    
                return np.reshape(M,[s1[0]*s2[0]*s3[0],])
    def vm(v):
                V = np.reshape(v,[s1[0],s2[0],s3[0]])
                M = np.tensordot(V,bra.conj(),([0],[0]))
                M = np.tensordot(M,ope.conj(),([0,3],[0,2]))
                M = np.tensordot(M,ket,([0,3],[0,2]))
                    
                return np.reshape(M,[s1[1]*s2[1]*s3[1],])
                
    tm = LinearOperator([s1[0]*s2[0]*s3[0],s1[1]*s2[1]*s3[1]],matvec = mv,rmatvec = vm)
    return tm


def toreal(Tensor):
    if np.linalg.norm(Tensor.imag)/np.linalg.norm(Tensor.real)<=1e-10:
        
        return Tensor - Tensor.imag*1j
    else: 
        return Tensor



def col_contract33_sparse(T1,T3):
    #T1,T2 3th order tensor   
    #--T1--
    #  |      -->  --T-- (linear operator)
    #--T3--
    bra = T1
    ket = T3
    
    s1 = np.shape(bra) 
    s3 = np.shape(ket) 
    def mv(v):
                V = np.reshape(v,[s1[1],s3[1]])
                M = np.tensordot(bra,V,([1],[0]))
                M = np.tensordot(M,ket.conj(),([1,2],[2,1]))
                    
                return np.reshape(M,[s1[0]*s3[0],])
    def vm(v):
                V = np.reshape(v,[s1[0],s3[0]])
                M = np.tensordot(V,bra.conj(),([0],[0]))
                M = np.tensordot(M,ket,([0,2],[0,2]))
                    
                return np.reshape(M,[s1[1]*s3[1],])
                
    tm = LinearOperator([s1[0]*s3[0],s1[1]*s3[1]],matvec = mv,rmatvec = vm)
    return tm
        


def entropy(s):
    assert np.linalg.norm(s.imag)<1e-10, f'The Schmidt values should be real {s}'        
    s = s/np.linalg.norm(s)
    Sa  = -np.sum(s**2*np.log(s**2))
    return Sa

def sqrthm(A,threshold=1e-10,max_bond=1000):
    #sqrt of the hermitian matrix A
    #discard the eigenvalues that smaller than threshold
    #XX^dag = A

    lam,vr = np.linalg.eigh(A)
    idx = lam.argsort()[::-1]
    if lam[idx[0]]<threshold and abs(lam[idx[-1]])>threshold:
        lam,vr = np.linalg.eigh(-A)
        idx = lam.argsort()[::-1]

    
    dim = np.sum(lam[idx]>threshold)
    dim = min(dim,max_bond)
    sqrtlam = np.sqrt(lam[idx[0:dim]])
    X = vr[:,idx[:dim ]]@np.diag(sqrtlam)
    assert dim >0, 'positive dimension expected '+f"lam = {lam} "
    return X


def find_phase(M):
    s = M.shape
    for i in range(s[0]):
        if abs((M[i,i])) > 1e-4:
            theta = M[i,i]/abs(M[i,i])
            return theta
    for i in range(s[0]):
        for j in range(i,s[1]):
            if abs(np.real(M[i,j])) > 1e-4:
                theta = (M[i,j]/(M[j,i].conj()))**0.5
                return theta
def is_hermitian_upto_a_phase(M):
    
    s = M.shape
    if s[0]!=s[1]:
        return False
        
    norm = np.linalg.norm(M)
    if norm<1e-10:
        return True
    
    theta = find_phase(M)
    
    M = M/theta
    norm = np.linalg.norm(M-M.transpose().conj())
    
    """N = M+M.transpose()
    if np.linalg.norm(np.real(N))>=100*np.linalg.norm(np.imag(N)):
        return True
    else:
        return False"""
    
    if norm*abs(theta)<1e-2*np.linalg.norm(M):
        return True
    else:
        print([norm,abs(theta),norm*abs(theta)])
        return False              


def delta_tensor(N,m):
    #return the N-th order delta tensor
    T = np.zeros([m]*N)
    for i in range(m):
        
        T[tuple([i]*N)] = 1
    return T

def single_trans(h1,h2,u1="I",u2="I"):
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    eZ = linalg.expm(np.array([[h1/2,0],[0,-h1/2]]))
    eX = linalg.expm(np.array([[0,h2/2],[h2/2,0]]))
    
    P = np.tensordot(P,eZ,([2],[0]))
    P = np.tensordot(P,eX,([2],[0]))
    
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


def single_T(h1,h2,u1="I",u2="I"):
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    eZ = linalg.expm(np.array([[h1/2,0],[0,-h1/2]]))
    eX = linalg.expm(np.array([[0,h2/2],[h2/2,0]]))
    
    P = np.tensordot(P,eZ,([2],[0]))
    P = np.tensordot(P,eX,([2],[0]))
    
    matrix_list={"I":np.eye(2),"X":np.matrix([[0,1],[1,0]])} 
    u1 = matrix_list[u1]
    u2 = matrix_list[u2]

    A3 = np.tensordot(P,A,([1],[0]))
    A3 = np.transpose(A3,[1,0,2,3,4])
    A3 = np.reshape(A3,[2,2,2,2,2])
    
    A31 = np.tensordot(A3,u1,([4],[0]))
    A32 = np.tensordot(A3,u2,([4],[0]))
    dA3 = np.tensordot(A31,A32,([0],[0]))
    dA3 = np.transpose(dA3,[0,4,1,5,2,6,3,7])
    dA3 = np.reshape(dA3,[4,4,4,4])
    return dA3


def single_trans_dephasing(p):
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    
    M = np.zeros([2,2,3])
    M[:,:,0] = np.eye(2)*np.sqrt(1-p)
    M[:,:,1] = np.array([[np.sqrt(p),0],[0,0]])
    M[:,:,2] = np.array([[0,0],[0,np.sqrt(p)]])
    
    dM = np.tensordot(M,M,([2],[2]))
    
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM = np.tensordot(A3,dM,([5],[0]))
    dA3dM = np.tensordot(A3,A3dM,([5],[6]))
    dA3ddM = np.tensordot(dA3dM,dM,([4,9],[1,3]))

    ddA3ddM = np.tensordot(dA3ddM,dA3ddM,([8,9,10,11],[9,8,11,10]))
    
    ddA3ddM = np.transpose(ddA3ddM,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    ddA3ddM = ddA3ddM.reshape([16,16,16,16])

    return ddA3ddM

def single_trans_qc(p,channel = 'dephasing'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    assert channel == 'dephasing' or channel == 'depolarizing' or channel == 'deamp', 'quantum channel should be dephasing, depolarizing or deamp' 
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    if channel == 'dephasing':
        M = np.zeros([2,2,3])
        M[:,:,0] = np.eye(2)*np.sqrt(1-p)
        M[:,:,1] = np.array([[np.sqrt(p),0],[0,0]])
        M[:,:,2] = np.array([[0,0],[0,np.sqrt(p)]])
    elif channel == 'depolarizing':
        M = np.zeros([2,2,4],dtype = 'complex')        
        M[:,:,0] = np.eye(2)*np.sqrt(1-p)
        M[:,:,1] = np.sqrt(p/3) * np.array([[1,0],[0,-1]])
        M[:,:,2] = np.sqrt(p/3) * np.array([[0,1],[1,0]])
        M[:,:,3] = np.sqrt(p/3) * np.array([[0,-1j],[1j,0]])
        
    elif channel == 'deamp':
        M = np.zeros([2,2,2])
        M[:,:,0] =  np.array([[1,0],[0,np.sqrt(1-p)]])
        M[:,:,1] = np.array([[0,np.sqrt(p)],[0,0]])
    
    dM = np.tensordot(M,M.conj(),([2],[2]))
    
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM = np.tensordot(A3,dM,([5],[1]))
    dA3dM = np.tensordot(A3,A3dM,([5],[7]))
    dA3ddM = np.tensordot(dA3dM,dM,([4,9],[1,3]))

    ddA3ddM = np.tensordot(dA3ddM,dA3ddM,([8,9,10,11],[9,8,11,10]))
    
    ddA3ddM = np.transpose(ddA3ddM,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    ddA3ddM = ddA3ddM.reshape([16,16,16,16])

    return ddA3ddM



def single_T_deamp(p):
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    
    M = np.zeros([2,2,3])
    M[:,:,0] = np.array([[1,0],[0,np.sqrt(1-p)]])
    M[:,:,1] = np.array([[0,np.sqrt(p)],[0,0]])
    
    dM = np.tensordot(M,M,([1,2],[1,2]))
    
    ddM = np.kron(dM,dM)
    ddM = ddM.reshape([4,4])

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    A3 = np.reshape(A3,[2,2,2,2,4])
    
    A3 = np.tensordot(A3,ddM,([4],[0]))
 

    A3s = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3s = np.transpose(A3s,[0,2,3,4,1,5])
    A3s = np.reshape(A3s,[2,2,2,2,4])
    
    dA3 = np.tensordot(A3,A3s,([4],[4]))
    dA3 = np.transpose(dA3,[0,4,1,5,2,6,3,7])
    dA3 = np.reshape(dA3,[4,4,4,4])
    return dA3

def simplify_trans(T):
    s = T.shape
    T1 = T.reshape([s[0],s[1]*s[2]*s[3]])
    
    U1,S1,V1 = np.linalg.svd(T1)
    dim = np.sum(S1>1e-10)
    print(S1)
    U1 = U1[:,:dim]
    S1 = np.diag(S1[:dim])
    V1 = V1[:dim,:]
    
    T2 = S1@V1
    T2 = T2.reshape([dim,s[1],s[2],s[3]])
    
    T2 = np.tensordot(T2,U1,([1],[0]))
    T2 = np.transpose(T2,[0,3,1,2])
    return T2