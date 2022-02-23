from ._RL_agent import *

class IM_TDLR(RL_agent):
    '''Unlucky symbol incorrect model agent adapted from Silva & Hare 2020.
    Agent has seperate learning rates for second-step values following 
    transitions it belives are common vs rare..'''

    def __init__(self, kernels=['bs', 'ck']):
        self.name = 'IM_TDLR'
        self.param_names  = ['alpC', 'alpR', 'iTemp', 'alpT']
        self.param_ranges = ['unit', 'unit', 'pos'  , 'unit']
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpC, alpR, iTemp, alpT = params_T[:4]   

        # Variables.
        V = np.zeros([2,session.n_trials])  # Second step TD values.
        T = np.zeros([2,session.n_trials])  # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            # Update action values and transition probabilities.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.
            
            if int(T[c,i] > 0.5) == c: # Set learning rate based on common vs rare transition.
               alpV = alpC
            else:
                alpV = alpR

            V[r,i+1] = V[r,i]
            T[n,i+1] = T[n,i] 

            V[s,i+1] = (1.-alpV)*V[s,i] + alpV*o # Second step TD update.
            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net = self.apply_kernels(M, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net, iTemp)