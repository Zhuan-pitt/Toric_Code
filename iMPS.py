import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg
import funcs


class iMPS(object):
    """
    Args:
        L : int, length of the unit cell
        B: list of rank 3 numpy tensors
        BC: boudary conditions, "PBC" or "OPC"
        s: schmidt weight at each bond, list of numpy vectors 
        chi: bond dimension at each bond, array
        d: list of physical dimension
        dtype: type of the tensors, float or complex
    """
    # (0)--T-- (1)
    #   (2)|   
    
    def check_consistency(self):
        L = self.L
        d = self.d
        assert isinstance(self.L,int)
        assert len(self.B) == self.L
        assert len(self.d) == self.L
        assert isinstance(d, np.ndarray) and d.shape == (L,) 
        assert len(self.chi) == self.L
        assert len(self.s) == self.L
        
        
        
        for i in range(self.L):
            assert isinstance(self.B[i], np.ndarray)
            assert isinstance(self.s[i], np.ndarray)
            assert self.chi[i]>0
            assert self.d[i]>0
            
            assert np.shape(self.B[i]) == ((self.chi[i],self.chi[(i+1)%L],self.d[i]))
                
            
            
        
    def construct_from_tensor_list(self,tensor_list):
        self.B=[]
        for i in range(len(tensor_list)):
                self.B.append(tensor_list[i])
        self.L = len(self.B)
        self.s = [None]*self.L
        self.chi = np.zeros(self.L,dtype = int)
        self.d = np.zeros(self.L,dtype = int)
        for i in range(self.L):
            self.chi[i] = (self.B[i].shape)[0]
            self.d[i] = (self.B[i].shape)[2]
            self.s[i] = np.eye(self.chi[i])
        self.canonical()
        self.dtype = self.B[0].dtype  
        self.check_consistency()
    
        
    def transfer_matrix(self,site=0):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix at given site
        """
        tensor = self.B[site] 
        s = np.shape(tensor) 
        
        if self.chi[site]>4:          
            def mv(v):
                V = np.reshape(v,[s[1],s[1]])
                M = np.tensordot(np.tensordot(tensor,V,([1],[0])),tensor.conj(),([1,2],[2,1]) )
                return np.reshape(M,[s[1]**2,])
            def vm(v):
                V = np.reshape(v,[s[0],s[0]])
                M = np.tensordot(np.tensordot(tensor.conj(),V,([0],[0])),tensor,([1,2],[2,0]) )
                return np.reshape(M,[s[1]**2,])
            
            tm = LinearOperator([s[0]**2,s[1]**2],matvec = mv, rmatvec = vm)
        else:
            tm = funcs.col_contract33(tensor,tensor)
        
        return tm
        
        
        
        
    def gram_matrix(self,site,direction = 'right'):
        """calculating the right/left gram matrix of the transfer matrix
        Args:
            site:  int, position of the transfer matrix
            direction: str, right or left gram matrix. Defaults to 'right'.
        """
        assert direction == 'right' or 'left', 'right or left gram matrix expected'
        
        trans = self.transfer_matrix(site)
        if direction == 'right':
            if isinstance(trans,linalg.LinearOperator):
                lam,v = linalg.eigs(trans,2)
            else:
                lam,v = np.linalg.eig(trans)
        if direction == 'left':
            if isinstance(trans,linalg.LinearOperator):
                lam,v = linalg.eigs(trans.adjoint(),2)
            else:
                lam,v = np.linalg.eig(trans.conj().transpose())        
                
        
        idx = lam.argsort()[::-1]
        self.B[site] = self.B[site]/np.sqrt(abs(lam[idx[0]]))
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        v = v+v.transpose().conj()
        v = v/np.linalg.norm(v)
        
        return v
        
        
        
    def canonical(self,max_bond=10,threshold=1e-10):
        """ transform the iMPS into the right canonical form: 
        . . . --G--G--- . . . 
                |  |             ,
        where 
        --G---     --         --s--G--     -s-
          |   |  =   |  and  |     |  =   |
        --G---     --         --s--G--     -s- .
        """
        for site in range(self.L):
            
            
            
            
            vr = self.gram_matrix(site,'right')
            vl = self.gram_matrix(site,'left')

            
            Mr = np.reshape(vr,[self.chi[site]]*2)
            Ml = np.reshape(vl,[self.chi[site]]*2)
            
            

            X = funcs.sqrthm(Mr,threshold)
            Y = funcs.sqrthm(Ml,threshold)
            

            
            U,s,V = np.linalg.svd(Y.transpose()@X)      
            dim = np.sum(s>threshold)
            
            
            assert dim >0, 'non-zero dimension expected'
            dim = min(max_bond,dim)
            
            U = U[:,:dim]
            s = s[:dim]
            V = V[:dim,:]
            
            s = s/np.linalg.norm(s)
            
            G = funcs.row_contract23(V@np.linalg.pinv(X),self.B[site])
            G = funcs.row_contract32(G, np.linalg.pinv(Y.transpose())@U@np.diag(s))
            
            self.B[site] = G
            self.chi[site] = dim
            self.s[site] = np.diag(s)
            
            assert self.L==1, 'need add code to deal with the case when unit cell larger than 1'
            #if self.L>1:
                
        
        
    
class iMPO:
    """
    Args:
        L : int, length of the unit cell
        B: list of rank 4 numpy tensors
        BC: boudary conditions, "PBC" or "OPC" 
        chi: bond dimension at each bond, array
        d: list of physical dimension
        dtype: type of the tensors, float or complex
    """
    #    (2)|   
    # (0) --T-- (1)
    #    (3)|   
    
        
    def check_consistency(self):
        assert isinstance(self.L,int)
        assert len(self.B) == self.L
        assert len(self.d) == self.L
        assert len(self.chi) == self.L
        assert self.chi.dtype == int or float
        assert self.d.dtype == int or float
        
        for i in range(self.L-1):
            assert isinstance(self.B[i], np.ndarray)
            assert self.B[i].dtype == self.dtype
            assert np.shape(self.B[i]) == ((self.chi[i],self.chi[i+1],self.d[i],self.d[i]))
        assert self.B[self.L-1].shape == ((self.chi[self.L-1],self.chi[0],self.d[self.L-1],self.d[self.L-1]))
            
            
            
            
        
    def construct_from_tensor_list(self,tensor_list):
        self.B=[]
        for i in range(len(tensor_list)):
                self.B.append(tensor_list[i])
        self.L = len(self.B)
        self.chi = np.zeros(self.L,dtype = int)
        self.d = np.zeros(self.L,dtype = int)
        for i in range(self.L):
            self.chi[i] = np.shape(self.B[i])[0]
            self.d[i] = np.shape(self.B[i])[2]
        self.dtype = tensor_list[0].dtype    
        
        self.check_consistency()
    
      

class MPS_power_method(object):
    """calculating the dominant eigenvector/eigenvalue of an MPO by using power method
    Args:
        Args:
        L : int, length of the unit cell
        max_bond: int, maximal bond dimension for each site

    """
    def __init__(self,MPS,MPO,max_bond):
        self.MPS = MPS
        self.MPS.canonical()
        self.MPO = MPO
        self.max_bond = max_bond
        self.E_history = []
        
    def update(self,site,loops):
        for _ in range(loops):
            B_new = funcs.col_contract34(self.MPS.B[site],self.MPO.B[site])
            self.MPS.B[site] = B_new
            self.MPS.chi[site] = B_new.shape[0]
            self.MPS.s[site] = np.kron(self.MPS.s[site],np.eye(self.MPO.chi[0]))
            self.MPS.canonical(self.max_bond)
            self.MPS.check_consistency()
            self.calculate_eig()
            if self.check_converge():
                return 
            
        
    def calculate_eig(self):
        
        s1 = np.shape(self.MPS.B[0])  
        s2 = np.shape(self.MPO.B[0])     
             
        site = 0
        if self.MPS.chi[site] >4:
            def mv(v):
                V = np.reshape(v,[s1[1],s2[1],s1[1]])
                M = np.tensordot(self.MPS.B[site],V,([1],[0]))
                M = np.tensordot(M,self.MPO.B[site],([2,1],[1,2]))
                M = np.tensordot(M,self.MPS.B[site].conj(),([3,1],[2,1]))
                
                return np.reshape(M,[s1[1]**2*s2[1],])
            
            trans = LinearOperator([s1[1]**2*s2[1]]*2,matvec = mv)
            E,_ = linalg.eigs(trans,1)
            idx = E.argsort()[::-1]
            E = E[idx[0]]

        else:
            trans = funcs.col_contract343(self.MPS.B[site],self.MPO.B[site],self.MPS.B[site])
            E = np.linalg.eigvals(trans)
            
            idx = E.argsort()[::-1]
            E = E[idx[0]]

        norm = np.linalg.norm(self.MPS.s[site])**2

        
        self.E_history.append(E/norm)
        
    def check_converge(self,threshold=1e-4,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False