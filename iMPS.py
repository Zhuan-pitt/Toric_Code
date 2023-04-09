import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg
import scipy
import funcs
import copy


def right2left(MPS):
    #transform a right canonical MPS to a left canonical MPS
    MPSl = copy.deepcopy(MPS)
    for i in range(MPS.L):
            MPSl.B[i] = funcs.row_contract23(np.diag(MPSl.s[i]),MPSl.B[i])
            MPSl.B[i] = funcs.row_contract32(MPSl.B[i],np.linalg.inv(np.diag(MPSl.s[(i+1)%MPS.L])))
    return MPSl

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

                
    def check_canonical_unit_cell(self):
        """
        check the canonical form of the unit cell
        """        
        
        trans = self.transfer_matrix('right')
        vr = np.eye(self.chi[0])
        vr = np.reshape(vr,[self.chi[0]**2,])
        V = trans.dot(vr)
        V = V.reshape([self.chi[0]]*2)
        V = V
        assert np.linalg.norm(V-V[0,0]*np.eye(self.chi[0]))<=1e-5*np.linalg.norm(V), \
            f'not right canonical, error = {repr(V)}'
            
            
            
        trans = self.transfer_matrix('left')
        vl = np.eye(self.chi[0])
        vl = np.reshape(vl,[self.chi[0]**2,])
        V = trans.rmatvec(vl.conj())
        V = V.reshape([self.chi[0]]*2).conj()
        V = V
        assert np.linalg.norm(V-V[0,0]*np.eye(self.chi[0])) <=1e-5*np.linalg.norm(V), \
            f'not left canonical, error = {repr(V)}'

        
    
    def chcek_canonical_site(self):
        for i in range(self.L-1,-1,-1):
            gammaB = self.B[i]
            transB = funcs.col_contract33(gammaB,gammaB)
            vr = np.eye(self.chi[(i+1)%self.L])
            vr = np.reshape(vr,[self.chi[(i+1)%self.L]**2,])
            V = transB.dot(vr)
            V = V.reshape([self.chi[i]]*2)
            assert np.linalg.norm(V-V[0,0]*np.eye(self.chi[i])) <=1e-5*np.linalg.norm(V), \
            f'not right canonical, error = {np.linalg.norm(V-V[0,0]*np.eye(self.chi[i]))}, site = {i}'
            
        
        for i in range(0,min(self.L,1)):
            
            B = self.B[0]
            s1= self.s[0]
            s2= self.s[1%self.L]

            B = funcs.row_contract32(funcs.row_contract23(np.diag(s1),B),np.linalg.inv(np.diag(s2)))
            
            trans = funcs.col_contract33_sparse(B,B)
            
            
            vl = np.eye(self.chi[0])
            vl = np.reshape(vl,[self.chi[0]**2,])
            
            V = trans.rmatvec(vl).conj()
            V = V.reshape([self.chi[1%self.L]]*2)
            

            assert np.linalg.norm(V-V[0,0]*np.eye(self.chi[1%self.L])) <=1e-5*np.linalg.norm(V),\
                f'not left canonical, error = {np.linalg.norm(V-V[0,0]*np.eye(self.chi[1%self.L]))}, site = {i}'
                
        
    def construct_from_tensor_list(self,tensor_list,threshold = 1e-10, max_bond = 20):
        self.B=[]
        for i in range(len(tensor_list)):
                self.B.append(tensor_list[i])
        self.L = len(self.B)
        self.s = [None]*self.L
        self.chi = np.zeros(self.L,dtype = int)
        self.d = np.zeros(self.L,dtype = int)
        self.svd_threshold=threshold
        self.max_bond = max_bond
        for i in range(self.L):
            self.chi[i] = (self.B[i].shape)[0]
            self.d[i] = (self.B[i].shape)[2]
            self.s[i] = np.ones([self.chi[i],])
        self.init_norm = self.calculate_norm()
        self.site_canonical(self.svd_threshold)
        self.dtype = self.B[0].dtype  
        
        
        self.check_canonical_unit_cell()
        self.chcek_canonical_site()

        
        
        self.check_consistency()        
        
    
    def clear_s(self):
        self.s[0] = np.ones([self.chi[0],])
            
    def construct_from_tensor_list_nocal(self,tensor_list):
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
            self.s[i] = np.ones([self.chi[i],])
        self.init_norm = self.calculate_norm()
        self.site_canonical(self.svd_threshold)
        self.dtype = self.B[0].dtype  
        self.check_consistency()
        #self.check_canonical_unit_cell()
    
    def single_matrix(self,site):
            
        
        tensor = self.B[site] 
        
        tm = funcs.col_contract33_sparse(tensor,tensor)
        
        return tm
    
    def transfer_matrix(self,direction='right'):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        if direction == 'right':
            matrix = self.single_matrix(0)
            for i in range(1,self.L):
                matrix = matrix.dot(self.single_matrix(i))
        if direction == 'left':   
            if self.L==1:     
                B1 = funcs.row_contract32(funcs.row_contract23(np.diag(self.s[0]),self.B[0]),np.linalg.inv(np.diag(self.s[0])))
                matrix = funcs.col_contract33_sparse(B1,B1)
            
            if self.L>1:
                B1 = funcs.row_contract23(np.diag(self.s[0]),self.B[0])
                matrix = funcs.col_contract33_sparse(B1,B1)
                for i in range(1,self.L-1):
                    B1 = self.B[i]
                    matrix = matrix.dot(funcs.col_contract33_sparse(B1,B1))
                    
                B1 = funcs.row_contract32(self.B[-1],np.linalg.inv(np.diag(self.s[0])))
                matrix = matrix.dot(funcs.col_contract33_sparse(B1,B1))
        return matrix
        
        
        
        
    def gram_matrix(self,direction = 'right'):
        """calculating the right/left gram matrix of the transfer matrix
        Args:
            site:  int, position of the transfer matrix
            direction: str, right or left gram matrix. Defaults to 'right'.
        """
        assert direction == 'right' or 'left', 'right or left gram matrix expected'
        
        trans = self.transfer_matrix(direction=direction)
        
        assert isinstance(trans,linalg.LinearOperator), 'wrong type'
        shape = trans.shape
        if direction == 'right':
            if shape[0]>3:
                #lam,v = np.linalg.eig(trans.dot(np.eye(shape[0])))
                lam,v = linalg.eigs(trans,2,v0 = np.ones([shape[1],]))
                
            else:
                lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        if direction == 'left':
            if shape[0]>3:
                #lam,v = np.linalg.eig(trans.adjoint().dot(np.eye(shape[0])))
                lam,v = linalg.eigs(trans.adjoint(),2,v0 = np.ones([shape[0],]))
            else:
                lam,v = scipy.linalg.eig(trans.adjoint() *  np.identity(shape[0])) 
                
        
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        
        v = v.reshape([self.chi[0]]*2)
        assert funcs.is_hermitian_upto_a_phase(v), print(repr(v))#'gram matrix should be hermitian'
        theta = funcs.find_phase(v)
        v=v/theta
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
                
                lam,v = linalg.eigs(trans,2,v0 = np.ones([shape[0],]))
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
        #self.normalize()
        
        Mr = self.gram_matrix('right')
        Ml = self.gram_matrix('left')
            
        
        X = funcs.sqrthm(Mr,threshold,self.max_bond)
        Y = funcs.sqrthm(Ml,threshold,self.max_bond)
            
        #Y,X = funcs.sqrt_left_right(Ml,Mr,self.s[0],max_bond = self.max_bond,threshold=threshold)
        s0 = np.diag(self.s[0])
        
        U,s,V = np.linalg.svd(Y.transpose().conj()@s0@X)      
            
        dim = len(s)
        assert dim >0, 'non-zero dimension expected'
        s = s/np.linalg.norm(s)
        
        #dim = np.sum(s>1e-10)
            
        U = U[:,:dim]
        s = s[:dim]
        V = V[:dim,:]
        
            
        s = s/np.linalg.norm(s)
        
        
        B0 = funcs.row_contract23(V@np.linalg.pinv(X),self.B[0])
        self.B[0] = B0
        
        B1 = funcs.row_contract32(self.B[-1], np.linalg.inv(s0)@np.linalg.pinv(Y.transpose().conj())@U@np.diag(s))
        self.B[-1] = B1
            
         
        self.chi[0] = dim
        self.s[0] = s
        self.normalize()
                
    def site_canonical(self,threshold = 1e-12):
        """generating the canonical form for each site"""
        self.clear_s()
        self.cell_canonical(threshold)
        
        if self.L>=2:        
            for i  in range(self.L-2,-1,-1):
                #self.single_site_svd(i,'right',threshold =threshold )
                self.two_site_svd(i)
                self.check_consistency()
                
        """self.clear_s()
        self.cell_canonical(threshold)
        
        if self.L>=2:        
            for i  in range(self.L-2,-1,-1):
                #self.single_site_svd(i,'right',threshold =threshold )
                self.two_site_svd(i)
                self.check_consistency()"""
                
        
        #self.clear_s()
        #self.cell_canonical(threshold)      
        
        
    def single_site_svd(self,site,direction = 'right',threshold = 1e-12):
        """applying svd on tensor at site,
        Args:
            site: int, site for the tensor
            direction: str, right or left canonical form
        """
        assert direction == 'right' or 'left', 'right or left gram matrix expected'
        #the iMPS is in right canonical form, therefore, self.s at i corresponds to the self.B at i-1
        s0 = np.diag(self.s[site])
        B1 = self.B[site]
        
        if direction =='right':
            B1  = B1.reshape([self.chi[site],self.chi[(site+1)%self.L]*self.d[site]])
        
            U,s,V = np.linalg.svd(B1)
            dim = np.sum(s>threshold)
            dim = min(self.max_bond,dim)
            U = U[:,:dim]
            V = V[:dim,:]
            s = s[:dim]
            U1 = U@np.diag(s)
            
            self.chi[(site)%self.L] = dim
            self.s[site] = s
            self.B[(site-1)%self.L] = funcs.row_contract32(self.B[(site-1)%self.L],U1)
            self.B[site] = V.reshape([dim,self.chi[(site+1)%self.L],self.d[site]])
        
        if direction == 'left':
            
            B1 = funcs.row_contract23(s0,B1)
            
            B1 = np.transpose(B1,[0,2,1])
            B1  = B1.reshape([self.chi[site]*self.d[site],self.chi[(site+1)%self.L]])
            
            U,s,V = np.linalg.svd(B1)
            
            
            dim = np.sum(s>threshold)
            dim = min(self.max_bond,dim)
            U = U[:,:dim]
            V = V[:dim,:]
            s = s[:dim]
            U1 = U@np.diag(s)
            U1 = U1.reshape([self.chi[site],self.d[site],dim])
            U1 = np.transpose(U1,[0,2,1])
            
            self.chi[(site+1)%self.L] = dim
            self.s[(site+1)%self.L] = s
            self.B[(site+1)%self.L] = funcs.row_contract23(V,self.B[(site+1)%self.L])
            self.B[site] = funcs.row_contract23(np.linalg.inv(s0),U1)

        
    def two_site_svd(self,site):
        """applying svd on tensor pair at site and site+1,
        Args:
            site: int, site for the first tensor
        """
        #the iMPS is in right canonical form, therefore, self.s at i corresponds to the self.B at i-1
        s0 = np.diag(self.s[site])
        B1 = self.B[site]
        B2 = self.B[(site+1)%self.L]
        merg_tensor = np.tensordot(funcs.row_contract23(s0,B1),B2,([1],[0]))
        
        merg_matrix = merg_tensor.reshape([self.chi[site]*self.d[site],self.chi[(site+2)%self.L]*self.d[(site+1)%self.L]])
        
        U,s,V = np.linalg.svd(merg_matrix)
        dim = np.sum(s>self.svd_threshold)
        dim = min(dim,self.max_bond)
        s = s[:dim]
        U = U[:,:dim]
        V = V[:dim,:]
        
        V = V.reshape([dim,self.chi[(site+2)%self.L],self.d[(site+1)%self.L]])
        self.B[(site+1)%self.L] = V
        self.chi[(site+1)%self.L] = dim
        
        U1 = (U@np.diag(s)).reshape([self.chi[site],self.d[site],dim])
        U1 = funcs.row_contract23(np.linalg.inv(s0),U1)
        self.B[site] = np.transpose(U1,[0,2,1])
        self.s[(site+1)%self.L] = s


    
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
    

        
class strap(object):
    
    
    def __init__(self,MPS1,MPO,MPS2):
        self.MPS1 = copy.deepcopy(MPS1)
        if MPO == None:
            self.MPO = iMPO()
            Ts=[]
            for i in range(self.MPS1.L):
                d = self.MPS1.d[i]
                T = np.zeros([1,1,d,d])
                T[0,0,:,:] = np.eye(d)
                Ts.append(T)
            self.MPO.construct_from_tensor_list(Ts)
        else:
            self.MPO = copy.deepcopy(MPO)
        self.MPS2 = copy.deepcopy(MPS2)
        self.MPS2.site_canonical()
        self.check_consistency()
        
        
    def check_consistency(self):
        assert self.MPS1.L == self.MPS2.L
        assert self.MPS1.L == self.MPO.L
        self.MPS1.check_consistency()
        self.MPS2.check_consistency()

    def transfer_matrix(self):
        """
        Returns:
            matrix: a instance of LinearOperator, transfer matrix of the unit cell
        """
        
        B1 = self.MPS1.B[0]
        B2 = self.MPS2.B[0]
        matrix = funcs.col_contract343_sparse(B1,self.MPO.B[0],B2 )
        for i in range(1,self.MPS1.L):
            matrix = matrix.dot(funcs.col_contract343_sparse(self.MPS1.B[i],self.MPO.B[i],self.MPS2.B[i]))
                
        return matrix
    
    
    
    def calculate_eig(self):
        
        trans = self.transfer_matrix()
        if trans.shape[0]>2:
            E,_ = linalg.eigs(trans,1,v0 = np.ones([trans.shape[1],]))
        else:
            E,_ = scipy.linalg.eig(trans * np.identity(trans.shape[0]))
        idx = E.argsort()[::-1]
        E = E[idx[0]]

        return E**(1/self.MPS1.L)      

class MPS_power_method(object):
    """calculating the dominant eigenvector/eigenvalue of an MPO by using power method
    Args:
        Args:
        L : int, length of the unit cell
        max_bond: int, maximal bond dimension for each site

    """
    def __init__(self,MPS,MPO,max_bond):
        MPS.max_bond = max_bond
        self.MPS = MPS
        
        self.MPS.site_canonical()
        self.MPO = MPO
        self.max_bond = max_bond
        
        self.E_history = []
        
    def update(self,loops,threshold=1e-3):
        for _ in range(loops):
            for site in range(self.MPS.L):
                B_new = funcs.col_contract34(self.MPS.B[site],self.MPO.B[site])
                self.MPS.B[site] = B_new
                self.MPS.chi[site] = B_new.shape[0]
                self.MPS.s[site] = np.ones([self.MPS.chi[site],])
            
            self.MPS.site_canonical(self.MPS.svd_threshold)
            self.update_lam()
            
            
            self.MPS.check_consistency()
            #self.calculate_eig()
            if self.check_converge(threshold):
                return 
    
    def update_lam(self):
        strap1 = strap(self.MPS,self.MPO,self.MPS)
        strap2 = strap(self.MPS,None,self.MPS)
        
        self.E_history.append(abs(strap1.calculate_eig())/abs(strap2.calculate_eig()))
            
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False
        

class MPS_power_method_single(object):
    """calculating the dominant eigenvector/eigenvalue of an MPO by using power method
    Args:
        Args:
        L : int, length of the unit cell
        max_bond: int, maximal bond dimension for each site

    """
    def __init__(self,MPS,MPO,max_bond):
        self.MPS = MPS
        self.MPS.svd_threshold = 1e-10
        self.MPS.site_canonical()
        self.MPO = MPO
        self.max_bond = max_bond
        MPS.max_bond = max_bond
        self.E_history = []
    
        
    def update(self,loops):
        for i in range(loops):
            
            print(i)
            
            MPS_single_site = MPS_singlesite_update(self.MPS,self.MPO,self.max_bond)
        
            MPS_single_site.svd_threshold=1e-10
            MPS_single_site.MPS2.svd_threshold=1e-10   
            MPS_single_site.init_MPS2(self.max_bond)
                
            #MPS_two_site.MPS2 = copy.deepcopy(MPS)
            MPS_single_site.init_env()
            MPS_single_site.update_MPS2(loop=10)
            
            self.MPS = copy.deepcopy(MPS_single_site.MPS2)
            
            self.E_history.append(self.MPS.calculate_norm())
            self.MPS.site_canonical(self.MPS.svd_threshold)
            self.MPS.check_canonical_unit_cell()
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
 
 
 
class MPS_power_method_twosite(object): 
    """calculating the dominant eigenvector/eigenvalue of an MPO by using power method
    Args:
        Args:
        L : int, length of the unit cell
        max_bond: int, maximal bond dimension for each site

    """
    def __init__(self,MPS,MPO,max_bond):
        MPS.max_bond = max_bond
        self.MPS = copy.deepcopy(MPS)
        self.MPS2 =  copy.deepcopy(MPS)
        self.MPS.site_canonical()
        self.MPO = MPO
        self.max_bond = max_bond
        self.E_history = []
    
    
    def new_MPS_twosite(self,loop=30):
        #solve MPS2 = self.MPO@MPS1 by using two site update method
        MPS_two_site = MPS_twosite_update2(self.MPS2,self.MPO,self.max_bond)
        MPS_two_site.MPS2r = copy.deepcopy(self.MPS2)
        MPS_two_site.init_MPS2()        
        MPS_two_site.init_env()
        MPS_two_site.update_MPS2(loop)
        
        self.MPS2 =  copy.deepcopy(MPS_two_site.MPS2r)
        
    def update(self,loops,threshold=1e-3):
        for _ in range(loops):
            
            self.new_MPS_twosite(loops)
            
            self.MPS2.site_canonical(self.MPS2.svd_threshold)
            
            self.MPS2.check_consistency()
            self.update_lam()
            #self.calculate_eig()
            if self.check_converge(threshold):
                return 
    
    def update_lam(self):
        strap1 = strap(self.MPS2,self.MPO,self.MPS2)
        strap2 = strap(self.MPS2,None,self.MPS2)
        
        self.E_history.append(abs(strap1.calculate_eig())/abs(strap2.calculate_eig()))
     
    
            
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold*E_average:
                return True
            else: return False
        else: return False

class MPS_twosite_update(object):
    def __init__(self,MPS,MPO,max_bond):
        MPS.site_canonical()
        self.L = MPS.L
        self.MPS1 = MPS
        
        self.MPO = MPO
        self.max_bond = max_bond
        self.overs=[]
        self.El = [None]*self.L
        self.Er = [None]*self.L
        self.E_history = []
    
        
    def init_MPS2(self):
        self.MPS2 = iMPS()
        Bs=[]
        np.random.seed(1)
        for site in range(self.L):
            
            d = self.MPO.d[site]
            B = np.zeros([1,1,d],dtype = 'complex')
            
            B[0,0] =(np.random.random([1,1,d])).reshape([d,])
            Bs.append(B)
        self.MPS2.svd_threshold =1e-10
        self.MPS2.max_bond = self.max_bond
        self.MPS2.construct_from_tensor_list(Bs)
        
        self.MPS2.max_bond = self.max_bond
        self.MPS2.site_canonical()
    
    def init_env(self):
        trans = self.single_matrix(0)
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i))
        
        shape = trans.shape
        if shape[0]>2:
            lam,v = linalg.eigs(trans,2,v0 = np.ones([shape[1],]))
        else:
            lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        self.Er[0] = v/np.linalg.norm(v)
        
        trans = self.single_matrix(0,'sG')
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i,'sG'))
        
        if shape[0]>2:
            lam,v = linalg.eigs(trans.adjoint(),2,v0 = np.ones([shape[0],]))
        else:
            lam,v = scipy.linalg.eig(trans.adjoint() *  np.identity(shape[0])) 
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]       
        self.El[0] = v.conj()/np.linalg.norm(v)

        
        over = (np.dot(self.El[0].conj(),self.Er[0]))
        phase = over/abs(over)
        self.El[0] = self.El[0]*phase
        
        for site in range(0,self.L-1):
            trans = self.single_matrix(site,order = 'sG')
            new_El = trans.rmatvec(self.El[site].conj()).conj()
            self.El[(site+1)%self.L] = new_El/np.linalg.norm(new_El)
        for site in range(self.L-1,0,-1):
            trans = self.single_matrix(site)    
            new_Er = trans.matvec(self.Er[(site+1)%self.L])
            
            self.Er[site] = new_Er/np.linalg.norm(new_Er)

    
    def single_matrix(self,site,order = 'Gs'):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of single site
        """
        if order == 'Gs':
            bra = self.MPS1.B[site] 
            ket = self.MPS2.B[site]
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        elif order == 'sG':
            s10 = np.diag(self.MPS1.s[site])
            s11 = np.diag(self.MPS1.s[(site+1)%self.L])
            s20 = np.diag(self.MPS2.s[site])
            s21 = np.diag(self.MPS2.s[(site+1)%self.L])
            
            bra = funcs.row_contract23(s10,self.MPS1.B[site] )
            bra = funcs.row_contract32(bra,np.linalg.inv(s11))
            ket = funcs.row_contract23(s20,self.MPS2.B[site] )
            ket = funcs.row_contract32(ket,np.linalg.inv(s21))
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        return tm
    
    def transfer_matrix(self,starting_site=0):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        site = starting_site%self.L
        s1 = np.diag(self.MPS1.s[site])
        s2 = np.diag(self.MPS2.s[site])
        B1 = funcs.row_contract23(s1,self.MPS1.B[site])
        B2 = funcs.row_contract23(s2,self.MPS2.B[site])
        matrix = funcs.col_contract343_sparse(B1,self.MPO.B[site],B2)
        for i in range(1,self.MPS1.L):
            
            site = (starting_site+i)%self.L
            matrix = matrix.dot(self.single_matrix(site))
                
        return matrix   
    
    
    def new_tensor(self,site):

        E_right = self.Er[(site+2)%self.L]
        E_left = self.El[site]
        s0 = np.diag(self.MPS1.s[site])
        
        Mr = np.reshape(E_right,[self.MPS1.chi[(site+2)%self.L],self.MPO.chi[(site+2)%self.L],-1])
        Ml = np.reshape(E_left,[self.MPS1.chi[site],self.MPO.chi[site],-1])

        Ml = np.tensordot(Ml,funcs.row_contract23(s0,self.MPS1.B[site]),([0],[0]))
        Ml = np.tensordot(Ml,self.MPO.B[site],([0,3],[0,2]))
        
        Mr = np.tensordot(Mr,self.MPS1.B[(site+1)%self.L],([0],[1]))
        
        Mr = np.tensordot(Mr,self.MPO.B[(site+1)%self.L],([0,3],[1,2]))
        
        new_M = np.tensordot(Ml,Mr,([1,2],[1,2]))
        new_M = np.transpose(new_M.conj(),[0,2,1,3])
        return new_M
    
    
    
    def two_site_svd(self,site):
        threshold = self.MPS2.svd_threshold
        site = site%self.L
        new_M = self.new_tensor(site)
        new_M = np.transpose(new_M,[0,2,1,3])
        w = np.shape(new_M)
        s0 = np.diag(self.MPS2.s[site])
        new_M = new_M.reshape([w[0]*w[1],w[2]*w[3]])
            
        U,lam,V = np.linalg.svd(new_M)
        
        dim = np.sum(lam>threshold)
        dim = min(dim,self.max_bond)
        self.MPS2.chi[(site+1)%self.L] = dim
            
        U = U[:,:dim]
        lam = lam[:dim]
        V = V[:dim,:]
        lam = lam/np.linalg.norm(lam)
            
        U = np.reshape(U@np.diag(lam),[-1,self.MPS2.d[site],dim])
        U  = np.transpose(U ,[0,2,1])
        U  = funcs.row_contract23(np.linalg.inv(s0),U)
        
        V = np.reshape(V,[dim,-1,self.MPS2.d[site]])

        return U,lam,V
    
    
    def update_env(self,site):
        # after updating the tensors at site and site+1
        trans = self.single_matrix(site,order = 'sG')
        new_El = (trans.rmatvec(self.El[site].conj())).conj()
        self.El[(site+1)%self.L] = new_El/np.linalg.norm(new_El)
        

        trans = self.single_matrix((site+1)%self.L)
        new_Er = trans.matvec(self.Er[(site+2)%self.L])
        self.Er[(site+1)%self.L] = new_Er/np.linalg.norm(new_Er)

    
    def cell_svd_update(self,site):
        site = site%self.L
        U,lam,V = self.two_site_svd(site)
            
        self.MPS2.B[site] = U
        self.MPS2.B[(site+1)%self.L] = V
        self.MPS2.s[(site+1)%self.L] = lam
        #if site != self.L-1:
        self.update_env(site)    
        
        
    
    def update_MPS2(self,loop=100):
        
        self.cell_svd_update(0)
        self.overs.append( self.overlap(0))
        for _ in range(loop):
            
            for site in range(1,self.L):
                self.cell_svd_update(site)
            for site in range(self.L-2,-1,-1):
                self.cell_svd_update(site)
                self.overs.append( self.overlap(site))
            if  self.check_list_converge(self.overs):
                break
            
        self.MPS2.site_canonical()
        #self.init_env()

        
    def overlap(self,site=0):
        trans = self.transfer_matrix(starting_site=site)
        over = trans.dot(self.Er[site])
        over = np.dot(self.El[site].conj(),over)
        
        return over
    
    def check_list_converge(self,over_list,threshold=1e-10):
        if len(over_list)>1:
            if abs(over_list[-1]-over_list[-2])<abs(threshold*over_list[-1]):
                    return True
            else: return False
        else: return False
        
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False
        


class MPS_twosite_update2(object):
    def __init__(self,MPS,MPO,max_bond):
        MPS.site_canonical()
        self.L = MPS.L
        self.MPS1r = copy.deepcopy(MPS)
        self.MPS1l = right2left(self.MPS1r)   
        self.MPO = MPO
        self.max_bond = max_bond
        self.overs=[]
        self.El = [None]*self.L
        self.Er = [None]*self.L
        self.E_history = []
        self.diff_list=[]
        
        
        
        Bs=[]
        np.random.seed(1)
        for site in range(self.L):
            
            d = self.MPO.d[site]
            B = np.zeros([1,1,d],dtype = 'complex')
            
            B[0,0] =(np.random.random([1,1,d])).reshape([d,])
            Bs.append(B)
        
        self.MPS2r = iMPS()
        self.MPS2r.svd_threshold = 1e-10
        self.MPS2r.max_bond = self.max_bond
        
        self.MPS2r.construct_from_tensor_list(Bs)
        
        self.svd_threshold =1e-12
        self.MPS2r.site_canonical()
    
        
    def init_MPS2(self):
        
        
        
        self.MPS2l = copy.deepcopy(self.MPS2r)
        
        
        for i in range(self.L):
            self.MPS2l.B[i] = funcs.row_contract23(np.diag(self.MPS2l.s[i]),self.MPS2l.B[i])
            self.MPS2l.B[i] = funcs.row_contract32(self.MPS2l.B[i],np.linalg.inv(np.diag(self.MPS2l.s[(i+1)%self.L])))
            
    
    def init_env(self):
        trans = self.single_matrix(0)
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i))
        
        
        shape = trans.shape
        if shape[0]>2:
            lam,v = linalg.eigs(trans,2,v0 = np.ones([shape[1],]))
        else:
            lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        v=v/v[0]   
        self.Er[0] = v/np.linalg.norm(v)
        
        trans = self.single_matrix(0,'sG')
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i,'sG'))
        
        if shape[0]>2:
            lam,v = linalg.eigs(trans.adjoint(),2,v0 = np.ones([shape[0],]))
        else:
            lam,v = scipy.linalg.eig(trans.adjoint() *  np.identity(shape[0])) 
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]   
        v=v/v[0]   
        self.El[0] = v.conj()/np.linalg.norm(v)

        
        
        
        for site in range(0,self.L-1):
            trans = self.single_matrix(site,order = 'sG')
            new_El = trans.rmatvec(self.El[site].conj()).conj()
            self.El[(site+1)%self.L] = new_El/np.linalg.norm(new_El)
        for site in range(self.L-1,0,-1):
            trans = self.single_matrix(site)    
            new_Er = trans.matvec(self.Er[(site+1)%self.L])
            self.Er[site] = new_Er/np.linalg.norm(new_Er)

    
    def single_matrix(self,site,order = 'Gs'):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of single site
        """
        if order == 'Gs':
            bra = self.MPS1r.B[site] 
            ket = self.MPS2r.B[site]
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        elif order == 'sG':
            bra = self.MPS1l.B[site] 
            ket = self.MPS2l.B[site]
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        return tm
    
    def transfer_matrix(self,starting_site=0):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        site = starting_site%self.L
        
            
        s1 = np.diag(self.MPS1l.s[(site+1)%self.L])
        s2 = np.diag(self.MPS2l.s[(site+1)%self.L])
        B1 = funcs.row_contract32(self.MPS1l.B[site],s1)
        B2 = funcs.row_contract32(self.MPS2l.B[site],s2)
        matrix = funcs.col_contract343_sparse(B1,self.MPO.B[site],B2)
    
        
        for i in range(1,self.L):
            
            site = (i+site)%self.L
            matrix = matrix.dot(self.single_matrix(site,order = 'Gs') )  
              
        return matrix   
    
    
    def new_tensor(self,site):

        E_right = self.Er[(site+2)%self.L]
        E_left = self.El[site]
        
        Mr = np.reshape(E_right,[self.MPS1r.chi[(site+2)%self.L],self.MPO.chi[(site+2)%self.L],-1])
        Ml = np.reshape(E_left,[self.MPS1r.chi[site],self.MPO.chi[site],-1])

        s1 = np.diag(self.MPS1r.s[(site+1)%self.L])
        Ml = np.tensordot(Ml,funcs.row_contract32(self.MPS1l.B[site],s1),([0],[0]))
        Ml = np.tensordot(Ml,self.MPO.B[site],([0,3],[0,2]))
        
        Mr = np.tensordot(Mr,self.MPS1r.B[(site+1)%self.L],([0],[1]))
        
        Mr = np.tensordot(Mr,self.MPO.B[(site+1)%self.L],([0,3],[1,2]))
        
        new_M = np.tensordot(Ml,Mr,([1,2],[1,2]))
        new_M = np.transpose(new_M,[0,2,1,3])
        return new_M
    
    
    
    def two_site_svd(self,site):
        threshold = self.svd_threshold
        site = site%self.L
        new_M = self.new_tensor(site)
        new_M = new_M/np.linalg.norm(new_M)
        
        new_M = np.transpose(new_M,[0,2,1,3])
        w = np.shape(new_M)
        new_M = new_M.reshape([w[0]*w[1],w[2]*w[3]])
            
        U,lam,V = np.linalg.svd(new_M)
        dim = np.sum(lam>threshold)
        dim = min(dim,self.max_bond)
        self.MPS2r.chi[(site+1)%self.L] = dim
        self.MPS2l.chi[(site+1)%self.L] = dim    
        U = U[:,:dim]
        lam = lam[:dim]
        V = V[:dim,:]
        lam = lam/np.linalg.norm(lam)
            
        U = np.reshape(U,[-1,self.MPS2r.d[site],dim])
        U  = np.transpose(U ,[0,2,1])
        #U  = funcs.row_contract23(np.linalg.inv(s0),U)
        
        V = np.reshape(V,[dim,-1,self.MPS2r.d[site]])

        return U,lam,V
    
    
    def update_env(self,site):
        # after updating the tensors at site and site+1
        trans = self.single_matrix(site,order = 'sG')
        new_El = (trans.rmatvec(self.El[site].conj())).conj()
        new_El = new_El/new_El[0]
        #new_El = new_El/np.linalg.norm(new_El)
        self.El[(site+1)%self.L] = new_El/np.linalg.norm(new_El)
        
        
        trans = self.single_matrix((site+1)%self.L)
        new_Er = trans.matvec(self.Er[(site+2)%self.L])
        new_Er = new_Er/new_Er[0]
        self.Er[(site+1)%self.L] = new_Er/np.linalg.norm(new_Er)
        

    
    def cell_svd_update(self,site):
        site = site%self.L
        U,lam,V = self.two_site_svd(site)
            
        self.MPS2l.B[site] = U
        self.MPS2r.B[(site+1)%self.L] = V
        self.MPS2l.s[(site+1)%self.L] = lam
        self.MPS2r.s[(site+1)%self.L] = lam
        
        self.update_env(site)    
        
        
    
    def update_MPS2(self,loop=100):
        for _ in range(loop):
            self.cell_svd_update(0)
            self.cell_svd_update(1)
            self.diff_list.append(self.difference())
            if self.diff_list[-1]<=1e-8:
                break
            
            if len(self.diff_list)>2:
                if  self.diff_list[-1]<1e-5 and self.diff_list[-1]>self.diff_list[-2]:
                    break
        


        
    def overlap(self,site=0):
        trans = self.transfer_matrix(starting_site=site)
        over = trans.dot(self.Er[site])
        over = np.dot(self.El[site].conj(),over)/(np.dot(self.El[site].conj(),self.Er[site]))
        
        return over
    
    def difference(self):
        s1 =np.diag(self.MPS2r.s[1])

        T1 = funcs.row_contract33(funcs.row_contract32(self.MPS2l.B[0],s1),self.MPS2r.B[1])
        
        T3 = self.new_tensor(0).conj()
        if T1.shape == T3.shape:
            return (np.linalg.norm(T1/T1[0,0,0,0]-T3/T3[0,0,0,0])/np.linalg.norm(T1/T1[0,0,0,0]))
        else:
            return 1000
        

    def check_list_converge(self,over_list,threshold=1e-10):
        if len(over_list)>1:
            if abs(over_list[-1]-over_list[-2])<abs(threshold*over_list[-1]):
                    return True
            else: return False
        else: return False
        
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False
        




class MPS_singlesite_update(object):
    def __init__(self,MPS,MPO,max_bond):
        MPS.site_canonical()
        self.L = MPS.L
        self.MPS1 = MPS
        self.MPS2 = iMPS()
        self.MPO = MPO
        self.max_bond = max_bond
        self.overs=[]
        self.El = [None]*self.L
        self.Er = [None]*self.L
        self.E_history = []
    
        
    def init_MPS2(self,dim=1):
        
        Bs=[]
        np.random.seed(1)
        for site in range(self.L):
            
            d = self.MPO.d[site]
            B = np.zeros([dim,dim,d],dtype = 'complex')
            
            B[:,:,:] =(np.random.random([dim,dim,d]))
            Bs.append(B)
        
        self.MPS2.construct_from_tensor_list(Bs)
        self.MPS2.svd_threshold =1e-10
        self.MPS2.max_bond = self.max_bond
        self.MPS2.site_canonical()
    
    def init_env(self):
        trans = self.single_matrix(0)
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i))
        
        shape = trans.shape
        if shape[0]>2:
            lam,v = linalg.eigs(trans,2,v0 = np.ones([shape[1],]))
        else:
            lam,v = scipy.linalg.eig(trans * np.identity(shape[0]))
            
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]
        self.Er[0] = v/np.linalg.norm(v)
        
        trans = self.single_matrix(0,'sG')
        for i in range(1,self.L):
            trans = trans.dot(self.single_matrix(i,'sG'))
        
        if shape[0]>2:
            lam,v = linalg.eigs(trans.adjoint(),2,v0 = np.ones([shape[0],]))
        else:
            lam,v = scipy.linalg.eig(trans.adjoint() *  np.identity(shape[0])) 
        idx = lam.argsort()[::-1]
        
        
        if len(idx)>=2:
            assert lam[idx[0]] != lam[idx[1]], 'nondegenerate state expected'
        v = v[:,idx[0]]       
        self.El[0] = v.conj()/np.linalg.norm(v)

        
        over = (np.dot(self.El[0].conj(),self.Er[0]))
        phase = over/abs(over)
        self.El[0] = self.El[0]*phase
        
        for site in range(0,self.L-1):
            trans = self.single_matrix(site,order = 'sG')
            new_El = trans.rmatvec(self.El[site].conj()).conj()
            self.El[(site+1)%self.L] = new_El/np.linalg.norm(new_El)
        for site in range(self.L-1,0,-1):
            trans = self.single_matrix(site)    
            new_Er = trans.matvec(self.Er[(site+1)%self.L])
            
            self.Er[site] = new_Er/np.linalg.norm(new_Er)

    
    def single_matrix(self,site,order = 'Gs'):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of single site
        """
        if order == 'Gs':
            bra = self.MPS1.B[site] 
            ket = self.MPS2.B[site]
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        elif order == 'sG':
            s10 = np.diag(self.MPS1.s[site])
            s11 = np.diag(self.MPS1.s[(site+1)%self.L])
            s20 = np.diag(self.MPS2.s[site])
            s21 = np.diag(self.MPS2.s[(site+1)%self.L])
            
            bra = funcs.row_contract23(s10,self.MPS1.B[site] )
            bra = funcs.row_contract32(bra,np.linalg.inv(s11))
            ket = funcs.row_contract23(s20,self.MPS2.B[site] )
            ket = funcs.row_contract32(ket,np.linalg.inv(s21))
            ope = self.MPO.B[site]
            tm = funcs.col_contract343_sparse(bra,ope,ket)
        return tm
    
    def transfer_matrix(self,starting_site=0):
        """
        Returns:
            tm: a instance of LinearOperator, transfer matrix of the unit cell
        """
        site = starting_site%self.L
        s1 = np.diag(self.MPS1.s[site])
        s2 = np.diag(self.MPS2.s[site])
        B1 = funcs.row_contract23(s1,self.MPS1.B[site])
        B2 = funcs.row_contract23(s2,self.MPS2.B[site])
        matrix = funcs.col_contract343_sparse(B1,self.MPO.B[site],B2)
        for i in range(1,self.MPS1.L):
            
            site = (starting_site+i)%self.L
            matrix = matrix.dot(self.single_matrix(site))
                
        return matrix   
    
    
    def new_tensor(self,site):

        E_right = self.Er[(site+1)%self.L]
        E_left = self.El[site]
        s0 = np.diag(self.MPS1.s[site])
        
        Mr = np.reshape(E_right,[self.MPS1.chi[(site+1)%self.L],self.MPO.chi[(site+1)%self.L],-1])
        Ml = np.reshape(E_left,[self.MPS1.chi[site],self.MPO.chi[site],-1])
        
        new_s = np.tensordot(s0,Mr,([1],[0]))
        
        new_s = np.tensordot(Ml,new_s,([0,1],[0,1]))
        
        
        new_s = new_s.conj()

        Ml = np.tensordot(Ml,funcs.row_contract23(s0,self.MPS1.B[site]),([0],[0]))
        Ml = np.tensordot(Ml,self.MPO.B[site],([0,3],[0,2]))
        
        
        new_B = np.tensordot(Ml,Mr,([1,2],[0,1]))
        new_B = np.transpose(new_B.conj(),[0,2,1])
        
        
        return new_B,new_s
    
    
    
    def site_update(self,site):
        site = site%self.L
        new_B,new_s = self.new_tensor(site)
        B = np.tensordot(np.linalg.inv(new_s),new_B,([1],[0]))
        self.MPS2.construct_from_tensor_list([B])
        self.init_env()
    
    
        
        
    
    def update_MPS2(self,loop=100):
        for _ in range(loop):
            self.site_update(0)
            self.overs.append( self.overlap(0))
            self.MPS2.site_canonical()
            
            if self.check_list_converge(self.overs):
                break
        #self.init_env()

        
    def overlap(self,site=0):
        trans = self.transfer_matrix(starting_site=site)
        over = trans.dot(self.Er[site])
        over = np.dot(self.El[site].conj(),over)
        
        return over
    
    def check_list_converge(self,over_list,threshold=1e-4):
        if len(over_list)>1:
            if abs(over_list[-1]-over_list[-2])<abs(threshold*over_list[-1]):
                    return True
            else: return False
        else: return False
        
    def check_converge(self,threshold=1e-3,loop=10):
        if len(self.E_history)>loop+1:
            E_average = np.mean(self.E_history[-loop-1:-1])
            if abs(E_average-self.E_history[-1])<threshold:
                return True
            else: return False
        else: return False
        
