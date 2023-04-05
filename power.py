import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import linalg
import scipy
import funcs
import copy
import iMPS

class MPS_power_method_singlesite_update(object):
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
        self.MPS2 = iMPS.iMPS()
        Bs=[]
        np.random.seed(1)
        for site in range(self.L):
            
            d = self.MPO.d[site]
            B = np.zeros([1,1,d],dtype = 'complex')
            
            B[0,0] =(np.random.random([1,1,d])).reshape([d,])
            Bs.append(B)
        
        self.MPS2.construct_from_tensor_list(Bs)
        self.MPS2.svd_threshold =1e-12
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
        
        """trans = self.single_matrix((site+1)%self.L)

        new_El = (trans.rmatvec(self.El[(site+1)%self.L].conj())).conj()
        self.El[(site+2)%self.L] = new_El/np.linalg.norm(new_El)"""
        
        trans = self.single_matrix((site+1)%self.L)
        new_Er = trans.matvec(self.Er[(site+2)%self.L])
        self.Er[(site+1)%self.L] = new_Er/np.linalg.norm(new_Er)
        
        """trans = self.single_matrix(site)
        new_Er = trans.matvec(self.Er[(site+1)%self.L])
        self.Er[site] = new_Er/np.linalg.norm(new_Er)"""
    
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
        
