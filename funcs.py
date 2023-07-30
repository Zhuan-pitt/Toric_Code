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

def field(f):
    if f =='X':
        return np.array([[0,1],[1,0]])
    if f=='Z':
        return np.array([[1,0],[0,-1]])
    if f=='I':
        return np.array([[1,0],[0,1]])


def single_trans_2layers_ff(h):
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    Id = np.eye(2)
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    A3 = np.reshape(A3,[2,2,2,2,4])
    
    dA3 = np.tensordot(A3,A3,axes=0)
    dA3 = np.transpose(dA3,[0,1,2,3,5,6,7,8,4,9])
    dA3 = np.reshape(dA3,[2,2,2,2,2,2,2,2,16])
    
    M = linalg.expm(h/2*np.kron(X@Z,np.kron(X@Z,np.kron(X@Z,X@Z))))
    
    #M=M@linalg.expm(h/2*np.kron(Z,np.kron(X,np.kron(Z,X))))
    
    dA3M = np.tensordot(dA3,M,([8],[0]))
    ddA3dM = np.tensordot(dA3M,dA3M,([8],[8]))
    ddA3dM=np.transpose(ddA3dM,[0,4,12,8,1,5,13,9,2,6,14,10,3,7,15,11])
    ddA3dM = ddA3dM.reshape([16]*4)
    return ddA3dM
    """X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    
    
    M = linalg.expm(np.kron(X@Z,Z@X)*h/2)
    
    M = M.reshape([2,2,2,2])

    
    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])

    
    dA3 = np.tensordot(A3,A3,axes=0)
    
    dA3M = np.tensordot(dA3,M,([5,11],[0,1]))
    
    dA3dM = np.tensordot(dA3M,M,([4,9],[0,1]))
   
    ddA3dM = np.tensordot(dA3dM,dA3dM,([8,9,10,11],[8,9,10,11]))
   
    
    ddA3dM=np.transpose(ddA3dM,[0,4,12,8,1,5,13,9,2,6,14,10,3,7,15,11])
    
    ddA3dM = ddA3dM.reshape([16]*4)"""
    

    #return ddA3dM

def single_trans_2layers_df(h,f1="X",f2="Z",perm = True):
    #double field
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    
    
    M = linalg.expm(np.kron(field(f1),field(f2))*h/2)
    
    M = M.reshape([2,2,2,2])

    
    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])

    
    dA3 = np.tensordot(A3,A3,axes=0)
    
    dA3M = np.tensordot(dA3,M,([5,11],[0,1]))
    
    dA3dM = np.tensordot(dA3M,M,([4,9],[0,1]))
   
    ddA3dM = np.tensordot(dA3dM,dA3dM,([8,9,10,11],[8,9,10,11]))
   
    
    if perm:
        ddA3dM=np.transpose(ddA3dM,[0,7,15,8,1,6,14,9,2,4,12,10,3,5,13,11])
    else:
        ddA3dM=np.transpose(ddA3dM,[0,4,12,8,1,5,13,9,2,6,14,10,3,7,15,11])
    
    ddA3dM = ddA3dM.reshape([16]*4)
    

    return ddA3dM
    

def single_trans_1layer(p,channel = 'dephasing',site_operator = np.array([[0,1],[1,0]])):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    
    A = delta_tensor(4,2)
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M = qc(p,channel)
    
    dM = np.tensordot(M,M.conj(),([2],[2]))
    dM = np.tensordot(site_operator,dM,([1],[0]))
    dM = np.trace(dM,axis1 = 0, axis2 = 2)

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM = np.tensordot(A3,dM,([5],[0]))
    dA3dM = np.tensordot(A3dM,A3,([5],[5]))
    dA3ddM = np.tensordot(dA3dM,dM,([4,9],[0,1]))

    dA3ddM = np.transpose(dA3ddM,[0,4,1,5,2,6,3,7])
    dA3ddM = np.reshape(dA3ddM,[4,4,4,4])
  
    return dA3ddM


def single_trans_qc(p,channel = 'dephasing'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M = qc(p,channel)
    
    dM = np.tensordot(M,M.conj(),([2],[2]))
    
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM = np.tensordot(A3,dM,([5],[1]))
    dA3dM = np.tensordot(A3dM,A3,([7],[5]))
    dA3ddM = np.tensordot(dA3dM,dM,([4,11],[1,3]))

    ddA3ddM = np.tensordot(dA3ddM,dA3ddM,([10,4,11,5],[11,5,10,4]))
    
    ddA3ddM = np.transpose(ddA3ddM,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    ddA3ddM = ddA3ddM.reshape([16,16,16,16])

    return ddA3ddM


def qc(p,channel):
    """return the single site quantum channel

    Args:
        p (_type_): error rate
    """
    channel_list = ['dephasing','depolarizing','deamp',"x_flip",'z_flip','xzf_flip']
    assert channel in channel_list, f'quantum channel should be chosen from {channel_list}.' 
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
    
    elif channel == 'x_flip':
        M = np.zeros([2,2,2])
        M[:,:,0] =  np.array([[np.sqrt(1-p),0],[0,np.sqrt(1-p)]])
        M[:,:,1] = np.array([[0,np.sqrt(p)],[np.sqrt(p),0]])
        
    elif channel == 'z_flip':
        M = np.zeros([2,2,2])
        M[:,:,0] =  np.array([[np.sqrt(1-p),0],[0,np.sqrt(1-p)]])
        M[:,:,1] = np.array([[np.sqrt(p),0],[0,-np.sqrt(p)]])
        
    elif channel == 'y_flip':
        M = np.zeros([2,2,2],dtype='complex')
        M[:,:,0] =  np.array([[np.sqrt(1-p),0],[0,np.sqrt(1-p)]],dtype='complex')
        M[:,:,1] = np.array([[0,-1*np.sqrt(p)],[1*np.sqrt(p),0]],dtype='complex')
        
    elif channel == 'xzf_flip':
        M = np.zeros([4,4,2],dtype='complex')
        M[:,:,0] =  np.sqrt(1-p)*np.array(np.kron(np.eye(2),np.eye(2)),dtype='complex')
        M[:,:,1] =  np.sqrt(p)*np.array(np.kron(np.array([[1,0],[0,1]],dtype='complex'),np.array([[1,0],[0,-1]],dtype='complex')))
        #M[:,:,2] =  np.sqrt(p/2)*np.array(np.kron(np.array([[0,1],[1,0]],dtype='complex'),np.array([[0,1],[1,0]],dtype='complex')))
        #M[:,:,3] =  np.sqrt(p*(1-p))*np.array(np.kron(np.eye(2,dtype='complex'),np.array([[1,0],[0,-1]],dtype='complex')))
        #M[:,:,2] =  np.sqrt(p/2)*np.kron(np.array(np.array([[1,0],[0,-1]],dtype='complex')),np.array([[0,1],[1,0]],dtype='complex'))
    
    return M

def single_trans_qc2(p1,p2,channel1 = 'dephasing',channel2 = 'deamp'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M1 = qc(p1,channel1)
    M2 = qc(p2,channel2)
    
    dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
    dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
    
    dM = np.tensordot(dM2,dM1,([1,3],[0,2]))
    dM = np.transpose(dM,([0,2,1,3]))

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM = np.tensordot(A3,dM,([5],[1]))
    dA3dM = np.tensordot(A3,A3dM,([5],[7]))
    dA3ddM = np.tensordot(dA3dM,dM,([4,9],[1,3]))

    ddA3ddM = np.tensordot(dA3ddM,dA3ddM,([8,9,10,11],[9,8,11,10]))
    
    ddA3ddM = np.transpose(ddA3ddM,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    ddA3ddM = ddA3ddM.reshape([16,16,16,16])

    return ddA3ddM

def single_trans_2layers(p1,p2,channel1 = 'dephasing',channel2 = 'deamp'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    
    
    
    if channel1 != 'xzf_flip' and channel1 != 'test':
        M1 = qc(p1,channel1)
        dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
        A3dM1 = np.tensordot(A3,dM1,([5],[1]))
        dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
        dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))
    if channel1 == 'xzf_flip':
        M1 = qc(p1,channel1)
        dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
        A31 = A3.reshape([2,2,2,2,4])
        A3dM1 = np.tensordot(A31,dM1,([4],[1]))
        dA3dM1 = np.tensordot(A3dM1,A31,([6],[4]))
        #dA3dM1 = dA3dM1.reshape([2]*12)
        #dA3ddM1 = dA3dM1.transpose([0,1,2,3,4,6,8,9,10,11,5,7])
    if channel1 == 'test':    
        M1 = qc(0,'x_flip')
        dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
        A3dM1 = np.tensordot(A3,dM1,([5],[1]))
        dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
        M1 = qc(p1,'x_flip')
        dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
        dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))
        
    
    
    if channel2 != 'xzf_flip' and channel2 != 'test' :
        M2 = qc(p2,channel2)
        dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
        A3dM2 = np.tensordot(A3,dM2,([5],[1]))
        dA3dM2 = np.tensordot(A3dM2,A3,([7],[5]))
        dA3ddM2 = np.tensordot(dA3dM2,dM2,([4,11],[1,3]))
        ddA3ddM = np.tensordot(dA3ddM1,dA3ddM2,([10,4,11,5],[11,5,10,4]))
    if channel2 == 'xzf_flip':
        M2 = qc(p2,channel2)
        dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
        A31 = A3.reshape([2,2,2,2,4])
        A3dM2 = np.tensordot(A31,dM2,([4],[1]))
        dA3dM2 = np.tensordot(A3dM2,A31,([6],[4]))
        #dA3dM2 = dA3dM2.reshape([2]*12)
        #dA3ddM2 = dA3dM2.transpose([0,1,2,3,4,6,8,9,10,11,5,7])
        ddA3ddM = np.tensordot(dA3dM1,dA3dM2,([4,5],[5,4]))
    
    
    if channel2 == 'test':    
        M2 = qc(0,'x_flip')
        dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
        A3dM2 = np.tensordot(A3,dM2,([5],[1]))
        dA3dM2 = np.tensordot(A3dM2,A3,([7],[5]))
        M2 = qc(p2,'x_flip')
        dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
        dA3ddM2 = np.tensordot(dA3dM2,dM2,([4,11],[1,3]))
        ddA3ddM = np.tensordot(dA3ddM1,dA3ddM2,([10,4,11,5],[11,5,10,4]))
    #ddA3ddM = np.tensordot(dA3ddM1,dA3ddM2,([10,4,11,5],[11,5,10,4]))
    
    
    
    ddA3ddM = np.transpose(ddA3ddM,[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    ddA3ddM = ddA3ddM.reshape([16,16,16,16])

    return ddA3ddM


def single_trans_2layers_proj(p1,p2,channel1 = 'dephasing',channel2 = 'deamp'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    
    E = np.zeros([2,2,2,2,2])
    E[0,0,0,0,0] = 1/2 
    E[1,1,1,1,0] = 1/2
    E[0,0,0,0,1] = 1/2
    E[1,1,1,1,1] = -1/2
    
    c = np.zeros([2,2,2])
    c[0,:,:] = np.eye(2)
    c[1,:,:] = np.array([[0,1],[1,0]])
    
    
    C = np.tensordot(c,c,([2],[1]))
    C = np.transpose(C,[0,2,1,3])
    
    EC = np.tensordot(C,E,([1],[0]))
    ECC = np.tensordot(EC,C,([5],[1]))
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M1 = qc(p1,channel1)
    M2 = qc(p2,channel2)
    
    dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
    dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM1 = np.tensordot(A3,dM1,([5],[1]))
    dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
    dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))

    A3dM2 = np.tensordot(A3,dM2,([5],[1]))
    dA3dM2 = np.tensordot(A3dM2,A3,([7],[5]))
    dA3ddM2 = np.tensordot(dA3dM2,dM2,([4,11],[1,3]))

    dA3ddM2ECC  = np.tensordot(dA3ddM2,ECC,([11,5],[1,7]))
    
    dA3ddM2dECC  = np.tensordot(dA3ddM2ECC,ECC,([9,4,14],[2,8,5]))
    
    
    ddA3ddMdECC = np.tensordot(dA3ddM1,dA3ddM2dECC ,([10,4,11,5],[9,13,15,19]))
    
    
    
    ddA3ddMdECC = np.transpose(ddA3ddMdECC,[0,4,8,12,16,20,1,5,9,13,17,21,2,6,10,14,18,22,3,7,11,15,19,23])
    
    #ddA3ddMdECC = np.transpose(ddA3ddMdECC,[0,4,20,8,12,16,1,5,21,9,13,17,2,6,22,10,14,18,3,7,23,11,15,19])
    
    ddA3ddMdECC = ddA3ddMdECC.reshape([64,64,64,64])
    
    return ddA3ddMdECC



def single_trans_1layer_proj(p1,channel1 = 'dephasing'):
    """
    Args:
        p (_type_): error rate
        channel (str, optional): quantum channel, including dephasing, depolarizing, deamp. 
        Defaults to 'dephasing'.

    Returns:
        _type_: single transfer tensor
    """
    
    E = np.zeros([2,2,2,2,2])
    E[0,0,0,0,0] = 1/2 
    E[1,1,1,1,0] = 1/2
    E[0,0,0,0,1] = 1/2
    E[1,1,1,1,1] = -1/2
    
    c = np.zeros([2,2,2])
    c[0,:,:] = np.eye(2)
    c[1,:,:] = np.array([[0,1],[1,0]])
    
    
    C = np.tensordot(c,c,([2],[1]))
    C = np.transpose(C,[0,2,1,3])
    
    EC = np.tensordot(C,E,([1],[0]))
    ECC = np.tensordot(EC,C,([5],[1]))
    
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M1 = qc(p1,channel1)
    
    dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM1 = np.tensordot(A3,dM1,([5],[1]))
    dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
    dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))

    dA3ddM1ECC  = np.tensordot(dA3ddM1,ECC,([5,11],[1,7]))
    
    dA3ddM1dECC  = np.tensordot(dA3ddM1ECC,ECC,([4,9,14,11,16],[2,8,5,1,6]))
    
    dA3ddM1dECC = dA3ddM1dECC.transpose([0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15])
    dA3ddM1dECC = dA3ddM1dECC.reshape([16,16,16,16])
    return dA3ddM1dECC


def single_trans_3layers(p1,channel1,p2,channel2,p3,channel3):
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M1 = qc(p1,channel1)
    M2 = qc(p2,channel2)
    M3 = qc(p3,channel3)
    
    dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
    dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
    dM3 = np.tensordot(M3,M3.conj(),([2],[2]))

    
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM1 = np.tensordot(A3,dM1,([5],[1]))
    dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
    dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))
    
    A3dM2 = np.tensordot(A3,dM2,([5],[1]))
    dA3dM2 = np.tensordot(A3dM2,A3,([7],[5]))
    dA3ddM2 = np.tensordot(dA3dM2,dM2,([4,11],[1,3]))
    
    A3dM3 = np.tensordot(A3,dM3,([5],[1]))
    dA3dM3 = np.tensordot(A3dM3,A3,([7],[5]))
    dA3ddM3 = np.tensordot(dA3dM3,dM3,([4,11],[1,3]))

    ddA3ddM12 = np.tensordot(dA3ddM1,dA3ddM2,([5,11],[4,10]))
    
    dddA3ddM123 = np.tensordot(ddA3ddM12,dA3ddM3,([9,4,19,14],[11,5,10,4]))
    
    dddA3ddM123 = np.transpose(dddA3ddM123,[0,4,8,12,16,20,1,5,9,13,17,21,2,6,10,14,18,22,3,7,11,15,19,23])
    dddA3ddM123 = dddA3ddM123.reshape([64,64,64,64])

    return dddA3ddM123

def single_trans_3layers_test(p1,channel1,p2,channel2,p3,channel3):
    A = delta_tensor(4,2)
    
    P = np.zeros([2,2,2])
    P[0,0,0],P[1,1,0],P[0,1,1],P[1,0,1]=1,1,1,1
    
    M1 = qc(p1,channel1)
    M2 = qc(p2,channel2)
    M3 = qc(p3,channel3)
    
    dM1 = np.tensordot(M1,M1.conj(),([2],[2]))
    M11 = qc(p1,'z_flip')
    dM11 = np.tensordot(M11,M11.conj(),([2],[2]))
    
    dM1 = np.tensordot(dM11,dM1,([1,3],[0,2]))
    dM1 = dM1.transpose([0,2,1,3])
    
    dM2 = np.tensordot(M2,M2.conj(),([2],[2]))
    dM3 = np.tensordot(M3,M3.conj(),([2],[2]))

    dM2 = dM1
    dM3 = dM1
    

    A3 = np.tensordot(np.tensordot(P,A,([1],[0])),P,([4],[0]))
    A3 = np.transpose(A3,[0,2,3,4,1,5])
    

    
    A3dM1 = np.tensordot(A3,dM1,([5],[1]))
    dA3dM1 = np.tensordot(A3dM1,A3,([7],[5]))
    dA3ddM1 = np.tensordot(dA3dM1,dM1,([4,11],[1,3]))
    
    A3dM2 = np.tensordot(A3,dM2,([5],[1]))
    dA3dM2 = np.tensordot(A3dM2,A3,([7],[5]))
    dA3ddM2 = np.tensordot(dA3dM2,dM2,([4,11],[1,3]))
    
    A3dM3 = np.tensordot(A3,dM3,([5],[1]))
    dA3dM3 = np.tensordot(A3dM3,A3,([7],[5]))
    dA3ddM3 = np.tensordot(dA3dM3,dM3,([4,11],[1,3]))

    ddA3ddM12 = np.tensordot(dA3ddM1,dA3ddM2,([5,11],[4,10]))
    
    dddA3ddM123 = np.tensordot(ddA3ddM12,dA3ddM3,([9,4,19,14],[11,5,10,4]))
    
    dddA3ddM123 = np.transpose(dddA3ddM123,[0,4,8,12,16,20,1,5,9,13,17,21,2,6,10,14,18,22,3,7,11,15,19,23])
    dddA3ddM123 = dddA3ddM123.reshape([64,64,64,64])

    return dddA3ddM123

