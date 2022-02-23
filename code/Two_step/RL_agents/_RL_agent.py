import numpy as np
from numba import jit
from .. import utility as ut

def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > ut.log_max_float] = ut.log_max_float # Protection agairt overflow in exponential.    
    expQT = np.exp(QT)
    return expQT/expQT.sum()

def array_softmax(Q,T = None):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([2,n_trials])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    if T is None: # Use default temperature of 1.
        TdQ = Q[1,:]-Q[0,:]
    else: 
        TdQ = T*(Q[1,:]-Q[0,:])
    TdQ[TdQ > ut.log_max_float] = ut.log_max_float # Protection agairt overflow in exponential.    
    P[0,:] = 1./(1. + np.exp(TdQ))
    P[1,:] = 1. - P[0,:]
    return P

def choose(P):
    "Takes vector of probabilities P summing to 1, returr integer s with prob P[s]"
    return sum(np.cumsum(P)<np.random.rand(1))

def session_log_likelihood(choices, Q_net, iTemp = None):
    'Evaluate session log likelihood given choices, action values and softmax temp.'
    choice_probs = array_softmax(Q_net, iTemp)
    session_log_likelihood = np.sum(ut.log_safe(
                                choice_probs[choices,np.arange(len(choices))]))
    return session_log_likelihood

# -------------------------------------------------------------------------------------
# Base class
# -------------------------------------------------------------------------------------

class RL_agent:

    def __init__(self, kernels = None):
        if kernels:
            self.use_kernels = True
            self.name = self.name + ''.join(['_'+k for k in kernels])
            for k in kernels:
                assert k in ['bs','ck'], 'Kernel type not recognised.'
                self.param_names  += [k]
                self.param_ranges += ['unc']                 
        else:
            self.use_kernels = False
        self.n_params = len(self.param_names)
        self.param_range_inds = {pr: [i for i, r in enumerate(self.param_ranges) if r == pr]
                                 for pr in set(self.param_ranges)}
        self.calculates_gradient = False
        self.type = 'RL'

    def apply_kernels(self, Q_pre, choices, second_steps, params_T):
        '''Apply modifier to entire sessions Q values due to kernels. 
        Kernel types:
        bs - Bias high vs low.
        ck - Choice kernel..'''
        if not self.use_kernels: return Q_pre                
        p_names = self.param_names
        bias = params_T[p_names.index('bs')] if 'bs' in p_names else 0.
        ck   = params_T[p_names.index('ck')] if 'ck' in p_names else 0.
        kernel_Qs = np.zeros((2,len(choices)))
        kernel_Qs[1, :] += bias
        kernel_Qs[1,1:] += ck*(choices[:-1]-0.5)
        return Q_pre + kernel_Qs