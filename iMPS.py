import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg
import scipy
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
    max_bond = 10
    svd_threshold = 1e-10
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

                
    def check_canonical_unit_cell(self):
        """
        check the canonical form of the unit cell
        """        
        
        trans = self.transfer_matrix()
        vr = np.eye(self.chi[0])
        vr = np.reshape(vr,[self.chi[0]**2,])
        V = trans.dot(vr)
        assert np.linalg.norm(V-vr*V[0]) <=1e-12, f'not right canonical, error = {np.linalg.norm(V-vr)}'


        vl = self.s[0]@self.s[0].conj().transpose()
        vl = np.reshape(vl,[self.chi[0]**2,])
        V = trans.rmatvec(vl)
        
        assert np.linalg.norm(V-vl*V[0]/vl[0]) <=1e-12, f'not left canonical, error = {np.linalg.norm(V-vl)}'    
    
    def chcek_canonical_site(self):
        for i in range(self.L-1,-1,-1):
            gammaB = self.B[i]
            transB = funcs.col_contract33(gammaB,gammaB)
            vr = np.eye(self.chi[(i+1)%self.L])
            vr = np.reshape(vr,[self.chi[(i+1)%self.L]**2,])
            V = transB.dot(vr)
            assert np.linalg.norm(V-vr*V[0]) <=1e-12, f'not right canonical, error = {np.linalg.norm(V-vr)}, site = {i}'
        for i in range(0,self.L):
            gammaA = self.B[i]
            transA = funcs.col_contract33(gammaA,gammaA)
            vl = self.s[i]@self.s[i].conj().transpose()
            vl = np.reshape(vl,[self.chi[i]**2,])
            V = vl@transA
            
            vl2 = self.s[(i+1)%self.L]@self.s[(i+1)%self.L].conj().transpose()
            vl2 = np.reshape(vl2,[self.chi[(i+1)%self.L]**2,])
            assert np.linalg.norm(V-vl2*V[0]/vl2[0]) <=1e-12, f'not left canonical, error = {np.linalg.norm(V-vr)}, site = {i}'
                
        
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
        self.init_norm = self.calculate_norm()
        self.site_canonical()
        self.dtype = self.B[0].dtype  
        self.check_consistency()
        self.check_canonical_unit_cell()
        self.chcek_canonical_site()
    
        
    def transfer_matrix(self):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        
        def single_matrix(site):
            
        
            tensor = self.B[site] 
            s = np.shape(tensor) 
            def mv(v):
                    V = np.reshape(v,[s[1],s[1]])
                    M = np.tensordot(np.tensordot(tensor,V,([1],[0])),tensor.conj(),([1,2],[2,1]) )
                    return np.reshape(M,[s[0]**2,])
            def vm(v):
                    V = np.reshape(v,[s[0],s[0]])
                    M = np.tensordot(np.tensordot(tensor.conj(),V,([0],[0])),tensor,([1,2],[2,0]) )
                    return np.reshape(M,[s[1]**2,])
                
            tm = LinearOperator([s[0]**2,s[1]**2],matvec = mv, rmatvec = vm)
            return tm
        
        matrix = single_matrix(0)
        for i in range(1,self.L):
            matrix = matrix.dot(single_matrix(i))
                
        return matrix
        
        
        
        
    def gram_matrix(self,direction = 'right'):
        """calculating the right/left gram matrix of the transfer matrix
        Args:
            site:  int, position of the transfer matrix
            direction: str, right or left gram matrix. Defaults to 'right'.
        """
        assert direction == 'right' or 'left', 'right or left gram matrix expected'
        
        trans = self.transfer_matrix()
        
        assert isinstance(trans,linalg.LinearOperator), 'wrong type'
        shape = trans.shape
        if direction == 'right':
            if shape[0]>3:
                lam,v = linalg.eigs(trans,2)
            else:
                lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        if direction == 'left':
            if shape[0]>3:
                lam,v = linalg.eigs(trans.adjoint(),2)
            else:
                lam,v = scipy.linalg.eig(trans.adjoint() *  np.identity(shape[0])) 
                
        
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        
        v = v.reshape([self.chi[0]]*2)
        assert funcs.is_hermitian_upto_a_phase(v), print(repr(v))#'gram matrix should be hermitian'
        v = v+v.transpose().conj()
        v = v/np.linalg.norm(v)
        v = v/np.sign(np.sum(v))
        return v
        
    def normalize(self):
        norm = self.calculate_norm()
        for i in range(self.L):
            self.B[i] = self.B[i]/(norm**(1/self.L)) 
    
    
    def calculate_norm(self):
        trans = self.transfer_matrix()
        
        assert isinstance(trans,linalg.LinearOperator), 'wrong type'
        shape = trans.shape
        if shape[0]>3:
                lam,v = linalg.eigs(trans,2)
        else:
                lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        
        idx = lam.argsort()[::-1]
        
        return np.sqrt(abs(lam[idx[0]]))
        
    def cell_canonical(self,threshold=1e-12):
        """ transform the iMPS into the right canonical form: 
        . . . --G--G--- . . . 
                |  |             ,
        where 
        --G---     --         --s--G--     -s-
          |   |  =   |  and  |     |  =   |
        --G---     --         --s--G--     -s- .
        """
        Mr = self.gram_matrix('right')
        Ml = self.gram_matrix('left').conj()
            
        self.normalize()
        X = funcs.sqrthm(Mr,threshold)
        Y = funcs.sqrthm(Ml,threshold)
            
        #Y,X = funcs.sqrt_left_right(Ml,Mr,self.s[0],max_bond = self.max_bond,threshold=threshold)
            
        U,s,V = np.linalg.svd(Y.transpose()@X)      
            
        dim = len(s)
        assert dim >0, 'non-zero dimension expected'

            
        U = U[:,:dim]
        s = s[:dim]
        V = V[:dim,:]
            
        #s = s/np.linalg.norm(s)
            
        self.B[0] = funcs.row_contract23(V@np.linalg.pinv(X),self.B[0])
        self.B[-1] = funcs.row_contract32(self.B[-1], np.linalg.pinv(Y.transpose())@U@np.diag(s))
            
         
        self.chi[0] = dim
        self.s[0] = np.diag(s)
                
    def site_canonical(self,threshold = 1e-12):
        """generating the canonical form for each site"""
        self.cell_canonical(threshold)
        if self.L>=2:
            """for i in range(self.L-2,-1,-1):
                self.two_site_svd(i)
                self.check_consistency()"""
                
            for i  in range(self.L-1,0,-1):
                self.single_site_svd(i,'right',threshold =threshold )
                self.check_consistency()
                
            for i  in range(0,self.L-1):
                self.single_site_svd(i,'left',threshold )
                self.check_consistency()  
                
        self.normalize()



        
    def single_site_svd(self,site,direction = 'right',threshold = 1e-12):
        """applying svd on tensor at site,
        Args:
            site: int, site for the tensor
            direction: str, right or left canonical form
        """
        assert direction == 'right' or 'left', 'right or left gram matrix expected'
        #the iMPS is in right canonical form, therefore, self.s at i corresponds to the self.B at i-1
        s0 = self.s[site]
        B1 = self.B[site]
        
        if direction =='right':
            B1  = B1.reshape([self.chi[site],self.chi[(site+1)%self.L]*self.d[site]])
        
            U,s,V = np.linalg.svd(B1)
            dim = np.sum(s>threshold)
            U = U[:,:dim]
            V = V[:dim,:]
            s = s[:dim]
            U1 = U@np.diag(s)
            
            self.chi[(site)%self.L] = dim
            self.s[site] = np.diag(s)
            self.B[(site-1)%self.L] = funcs.row_contract32(self.B[(site-1)%self.L],U1)
            self.B[site] = V.reshape([dim,self.chi[(site+1)%self.L],self.d[site]])
        
        if direction == 'left':
            
            B1 = funcs.row_contract23(s0,B1)
            
            B1 = np.transpose(B1,[0,2,1])
            B1  = B1.reshape([self.chi[site]*self.d[site],self.chi[(site+1)%self.L]])
            
            U,s,V = np.linalg.svd(B1)
            
            
            dim = np.sum(s>threshold)
            U = U[:,:dim]
            V = V[:dim,:]
            s = s[:dim]
            U1 = U@np.diag(s)
            U1 = U1.reshape([self.chi[site],self.d[site],dim])
            U1 = np.transpose(U1,[0,2,1])
            
            self.chi[(site+1)%self.L] = dim
            self.s[(site+1)%self.L] = np.diag(s)
            self.B[(site+1)%self.L] = funcs.row_contract23(V,self.B[(site+1)%self.L])
            self.B[site] = funcs.row_contract23(np.linalg.inv(s0),U1)

        
    def two_site_svd(self,site):
        """applying svd on tensor pair at site and site+1,
        Args:
            site: int, site for the first tensor
        """
        #the iMPS is in right canonical form, therefore, self.s at i corresponds to the self.B at i-1
        s0 = self.s[site]
        B1 = self.B[site]
        B2 = self.B[(site+1)%self.L]
        merg_tensor = np.tensordot(funcs.row_contract23(s0,B1),B2,([1],[0]))
        
        merg_matrix = merg_tensor.reshape([self.chi[site]*self.d[site],self.chi[(site+2)%self.L]*self.d[(site+1)%self.L]])
        
        U,s,V = np.linalg.svd(merg_matrix)
        
        dim = len(s)
        V = V.reshape([dim,self.chi[(site+2)%self.L],self.d[(site+1)%self.L]])
        self.B[(site+1)%self.L] = V
        self.chi[(site+1)%self.L] = dim
        
        U1 = (U@np.diag(s)).reshape([self.chi[site],self.d[site],dim])
        U1 = funcs.row_contract23(np.linalg.inv(s0),U1)
        self.B[site] = np.transpose(U1,[0,2,1])
        self.s[(site+1)%self.L] = np.diag(s)


    
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
        self.MPS.svd_threshold = 1e-12
        self.MPS.site_canonical()
        self.MPO = MPO
        self.max_bond = max_bond
        MPS.max_bond = max_bond
        self.E_history = []
        
    def update(self,loops):
        for _ in range(loops):
            for site in range(self.MPS.L):
                B_new = funcs.col_contract34(self.MPS.B[site],self.MPO.B[site])
                self.MPS.B[site] = B_new
                self.MPS.chi[site] = B_new.shape[0]
                self.MPS.s[site] = np.kron(self.MPS.s[site],np.eye(self.MPO.chi[site]))
            self.E_history.append(self.MPS.calculate_norm())
            self.MPS.site_canonical(self.MPS.svd_threshold)
            
            self.MPS.check_consistency()
            #self.calculate_eig()
            if self.check_converge():
                return 
            
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False
        
        
class strap(object):
    
    
    def __init__(self,MPS1,MPO,MPS2):
        self.MPS1 = MPS1
        self.MPO = MPO
        self.MPS2 = MPS2
        self.check_consistency()
        
    def check_consistency(self):
        self.MPS1.L = self.MPS2.L
        self.MPS1.L = self.MPO.L



    def transfer_matrix(self):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        
        def single_matrix(site):
            
        
            bra = self.MPS1.B[site] 
            ket = self.MPS2.B[site]
            ope = self.MPO.B[site]
            s1 = np.shape(bra) 
            s2 = np.shape(ope) 
            s3 = np.shape(ket) 
            def mv(v):
                    V = np.reshape(v,[s1[1],s2[1],s3[1]])
                    M = np.tensordot(bra,V,([1],[0]))
                    M = np.tensordot(M,ope,([2,1],[1,2]))
                    M = np.tensordot(M,ket.conj(),([3,1],[2,1]))
                    
                    return np.reshape(M,[s1[0]*s2[0]*s3[0],])

                
            tm = LinearOperator([s1[0]*s2[0]*s3[0],s1[1]*s2[1]*s3[1]],matvec = mv)
            return tm
        
        matrix = single_matrix(0)
        for i in range(1,self.MPS1.L):
            matrix = matrix.dot(single_matrix(i))
                
        return matrix
    
    
    
    def calculate_eig(self):
        
        trans = self.transfer_matrix()
        if trans.shape[0]>2:
            E,_ = linalg.eigs(trans,1)
        else:
            E,_ = scipy.linalg.eig(trans * np.identity(trans.shape[0]))
        idx = E.argsort()[::-1]
        E = E[idx[0]]

        return E**(1/self.MPS1.L)