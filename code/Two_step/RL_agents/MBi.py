from ._RL_agent import *

class MBi(RL_agent):
    'Model based agent.'

    def __init__(self, kernels=['bs', 'ck']):
        self.name = 'MBi'
        self.param_names  = ['alpV', 'iTemp', 'alpT']
        self.param_ranges = ['unit', 'pos'  , 'unit']
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpV, iTemp, alpT = params_T[:3]   

        # Variables.
        V = np.zeros([2,session.n_trials])  # Second step TD values.
        T = np.zeros([2,session.n_trials])  # Transition probabilities.
        # Initialize first trial transition probabilities.
        if session.number > 1:
            T[:,0] = [0.2,0.8] if session.blocks['transition_states'][0] else [0.8,0.2]
        else:
             T[:,0] = [0.5,0.5]

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            # Update action values and transition probabilities.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            V[r,i+1] = V[r,i]
            T[n,i+1] = T[n,i] 

            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o # Second step TD update.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = self.apply_kernels(M, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net, iTemp)