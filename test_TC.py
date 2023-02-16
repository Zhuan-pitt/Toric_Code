import numpy as np
from matplotlib import pyplot
from scipy import linalg
import copy
import iMPS
from scipy.sparse import linalg
import funcs

def check_canonical(chain,site):
    trans = chain.transfer_matrix(site)
    vr = np.eye(chain.chi[(site+1)%chain.L])
    vr = np.reshape(vr,[chain.chi[(site+1)%chain.L]**2,])
    if isinstance(trans,linalg.LinearOperator):    
        V = trans.dot(vr)
    else: 
        V = trans@vr    
    assert np.linalg.norm(V-vr) <=1e-12, f'not right canonical, error = {np.linalg.norm(V-vr)}'
           
    vl = chain.s[(site-1)%chain.L]@chain.s[(site-1)%chain.L].conj().transpose().conj()
    vl = np.reshape(vl,[chain.chi[(site-1)%chain.L]**2,])
    if isinstance(trans,linalg.LinearOperator):    
        V = trans.rmatvec(vl)
    else: 
        V = vl.conj()@trans       
    assert np.linalg.norm(V-vl) <=1e-12, f'not left canonical, error = {np.linalg.norm(V-vr)}'
    


def check_eig():
    h=0
    trans = funcs.single_trans(h)
    MPO = iMPS.iMPO()
    MPO.construct_from_tensor_list([trans])

    B = np.zeros([1,1,4])
    B[0,0] =np.eye(2).reshape([4,])
    MPS = iMPS.iMPS()
    MPS.construct_from_tensor_list([B])
    MPS_power = iMPS.MPS_power_method(MPS,MPO,10)
    for _ in range(100):
        MPS_power.update(0,1)
        assert abs(MPS_power.E_history[-1]-2)<1e-10, f'eigenvalue {MPS.E_history[-1]} is got, expected to be 2'
    

if __name__ == "__main__":
    
    h=0
    trans = funcs.single_trans(h)
    MPO = iMPS.iMPO()
    MPO.construct_from_tensor_list([trans])

    B = np.zeros([1,1,4])
    B[0,0] =np.eye(2).reshape([4,])
    MPS = iMPS.iMPS()
    MPS.construct_from_tensor_list([B])
    MPS_power = iMPS.MPS_power_method(MPS,MPO,50)
    MPS_power.update(0,10)
    
    check_canonical(MPS,0)
    check_eig()
    
    print("Everything passed")