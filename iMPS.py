import numpy as np
from scipy.sparse.linalg import LinearOperator
import contract 


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
        
        for i in range(self.L):
            assert isinstance(self.B[i], np.ndarray)
            assert len(np.shape(self.B[i])) == 3
            assert self.B[i].dtype == self.dtype
            #assert isinstance(self.s[i], np.ndarray)
            if i >0:
                assert np.shape(self.B[i])[0] == np.shape(self.B[i-1])[1] 
                assert np.shape(self.B[i])[0] == self.chi[i]
            else:
                assert np.shape(self.B[i])[0] == np.shape(self.B[self.L-1])[1] 
                
            assert np.shape(self.B[i])[2] == self.d[i]
            
            
        
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
        
        return self.transfer_matrix_class(tensor)
        
    class  transfer_matrix_class(LinearOperator):
            def __init__(self,tensor_at_bond):
                s = np.shape(tensor_at_bond)
                self.dtype = tensor_at_bond.dtype
                self.shape = (s[0]**2,s[1]**2)
                self.matvec = self.mv(tensor_at_bond)
            
            def mv(self,tensor):
                def transfer_operator(v):
                    s = np.shape(tensor)
                    V = np.reshape(v,[s[1],s[1]])
                    M = np.zeros_like(V)
                    
                    for i in range(s[2]):
                        M += tensor[:,:,i]@V@tensor[:,:,i].transpose().conj()
                    return np.reshape(M,[s[1]*2,])
                return transfer_operator

    
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
        
        for i in range(self.L):
            assert isinstance(self.B[i], np.ndarray)
            assert len(np.shape(self.B[i])) == 4
            assert self.B[i].dtype == self.dtype
            #assert isinstance(self.s[i], np.ndarray)
            if i >0:
                assert np.shape(self.B[i])[0] == np.shape(self.B[i-1])[1] 
                assert np.shape(self.B[i])[0] == self.chi[i]
            else:
                assert np.shape(self.B[i])[0] == np.shape(self.B[self.L-1])[1] 
                
            assert np.shape(self.B[i])[2] == self.d[i]
            assert np.shape(self.B[i])[3] == self.d[i]
            
            
            
            
        
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
    
      