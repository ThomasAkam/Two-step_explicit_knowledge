from ._RL_agent import *

class MF(RL_agent):
    'Model-free agent.'

    def __init__(self, kernels=['bs', 'ck']):
        self.name = 'MF'
        self.param_names  = ['alp', 'iTemp', 'lbd']
        self.param_ranges = ['unit', 'pos' , 'unit' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alp, iTemp, lbd = params_T[:3]   # Q value decay parameter.

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            # Update action values. 

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            Q[n,i+1] = Q[n,i] 
            V[r,i+1] = V[r,i]

            Q[c,i+1] = (1.-alp)*Q[c,i] + alp*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alp)*V[s,i] + alp*o  # Second step TD update.

        # Evaluate net action values and likelihood. 

        Q_net = self.apply_kernels(Q, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net, iTemp)