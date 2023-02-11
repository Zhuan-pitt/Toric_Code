import numpy as np
from scipy.sparse.linalg import LinearOperator



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
        assert self.chi.dtype == int or float
        assert self.d.dtype == int or float
        
        
        for i in range(self.L-1):
            assert isinstance(self.B[i], np.ndarray)
            assert self.B[i].dtype == self.dtype
            
            assert np.shape(self.B[i]) == ((self.chi[i],self.chi[i+1],self.d[i]))
        assert np.shape(self.B[self.L-1]) == ((self.chi[self.L-1],self.chi[0],self.d[self.L-1]))
                
            
            
        
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
    
        
    def transfer_matrix(self,site=0):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix at given site
        """
        tensor = self.B[site]        
        def transfer_operator(v):
            s = np.shape(tensor)
            V = np.reshape(v,[s[1],s[1]])
            M = np.tensordot(np.tensordot(tensor,V,([1],[0])),tensor.conj(),([1,2],[2,1]) )
            return np.reshape(M,[s[1]*2,])
        s = np.shape(tensor)
        tm = LinearOperator([s[0]**2,s[1]**2],matvec = transfer_operator)
        
        return tm
        

    
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
    
      