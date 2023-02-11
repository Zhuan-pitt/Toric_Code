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
        assert isinstance(self.L,int)
        assert len(self.B) == self.L
        assert len(self.d) == self.L
        assert len(self.chi) == self.L
        assert len(self.s) == self.L
        assert self.chi.dtype == int 
        assert self.d.dtype == int 
        
        
        for i in range(self.L-1):
            assert isinstance(self.B[i], np.ndarray)
            assert isinstance(self.s[i], np.ndarray)
            assert self.B[i].dtype == self.dtype
            
            assert np.shape(self.B[i]) == ((self.chi[i],self.chi[i+1],self.d[i]))
        assert np.shape(self.B[self.L-1]) == ((self.chi[self.L-1],self.chi[0],self.d[self.L-1]))
                
            
            
        
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
        self.dtype = tensor_list[0].dtype    
        
        self.check_consistency()
    
        
    def transfer_matrix(self,site=0,direction = 'right'):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix at given site
        """
        G = self.B[site] 
        s = np.shape(G) 
        if direction =='right':
            tensor = funcs.row_contract32(G,self.s[site])
        else:
            tensor = funcs.row_contract23(self.s[site],G)
                  
        def mv(v):
            V = np.reshape(v,[s[1],s[1]])
            M = np.tensordot(np.tensordot(tensor,V,([1],[0])),tensor.conj(),([1,2],[2,1]) )
            return np.reshape(M,[s[1]*2,])
        def vm(v):
            V = np.reshape(v,[s[1],s[1]])
            M = np.tensordot(np.tensordot(tensor,V,([0],[0])),tensor.conj(),([1,2],[2,0]) )
            return np.reshape(M,[s[1]*2,])
        
        s = np.shape(tensor)
        tm = LinearOperator([s[0]**2,s[1]**2],matvec = mv, rmatvec = vm)
        
        return tm
        
    def canonical(self):
        """ transform the iMPS into the canonical form: 
        --G--s--G--s--
          |     |       ,
        where 
        --G--s--     --         --s--G--     --
          |     |  =   |  and  |     |  =   |
        --G--s--     --         --s--G--     -- .
        """
        for site in range(self.L):
            trans = self.transfer_matrix(site)
            lam,vr = linalg.eigs(trans,2)
            idx = lam.argsort()[::-1]
            assert lam[idx[0]] != lam[idx[1]]
            vr = vr[:,idx[0]]
            
            lam,vl = linalg.eigs(trans.transpose(),2)
            idx = lam.argsort()[::-1]
            assert lam[idx[0]] != lam[idx[1]]
            vl = vl[:,idx[0]]
            
            
            Mr = np.reshape(vr,[self.chi[site]]*2)
            Ml = np.reshape(vl,[self.chi[site]]*2)
            
            X = funcs.sqrthm(Mr)
            Y = funcs.sqrthm(Ml)
            
            
            U,s,V = np.linalg.svd(Y.transpose()@X)
            G = funcs.row_contract23(V@np.linalg.inv(X),self.B[site])
            G = funcs.row_contract32(G, np.linalg.inv(Y.transpose())@U)
            
            self.B[site] = G
            self.s[site] = s
        
        
    
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
        self.chi = np.zeros(self.L,)
        self.d = np.zeros(self.L,)
        for i in range(self.L):
            self.chi[i] = np.shape(self.B[i])[0]
            self.d[i] = np.shape(self.B[i])[2]
        self.dtype = tensor_list[0].dtype    
        
        self.check_consistency()
    
      