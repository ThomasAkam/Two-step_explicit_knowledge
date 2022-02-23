from ._RL_agent import *
import numpy as np

class MF_MB(RL_agent):
    'Mixture agent.'

    def __init__(self, kernels=['bs', 'ck']):
        self.name = 'MF_MB'
        self.param_names  = ['alpQ', 'lbd' , 'alpT', 'G_td', 'G_mb']
        self.param_ranges = ['unit', 'unit', 'unit', 'pos' , 'pos' ]
        RL_agent.__init__(self, kernels)

    @jit
    def session_likelihood(self, session, params_T):

        # Unpack trial events.
        choices, second_steps, outcomes = session.unpack_trial_data('CSO')

        # Unpack parameters.
        alpQ, lbd, alpT, G_td, G_mb= params_T[:5]

        #Variables.
        Q = np.zeros([2,session.n_trials]) # First step TD values.
        V = np.zeros([2,session.n_trials]) # Second step TD values.
        T = np.zeros([2,session.n_trials]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.

        for i, (c, s, o) in enumerate(zip(choices[:-1], second_steps, outcomes)): # loop over trials.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            # Update action values and transition probabilities.

            Q[n,i+1] = Q[n,i] 
            V[r,i+1] = V[r,i]
            T[n,i+1] = T[n,i]

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

        # Evaluate net action values and likelihood. 

        M = T*V[1,:] + (1.-T)*V[0,:] # Model based action values.
        Q_net =  G_td*Q + G_mb*M     # Mixture of model based and model free values.
        Q_net = self.apply_kernels(Q_net, choices, second_steps, params_T)

        return session_log_likelihood(choices, Q_net)


    def simulate(self, task, params_T, n_trials):

        # Unpack parameters.
        alpQ, lbd, alpT, G_td, G_mb, bs, ck = params_T

        #Variables.
        Q = np.zeros([2,n_trials+1]) # First step TD values.
        V = np.zeros([2,n_trials+1]) # Second step TD values.
        T = np.zeros([2,n_trials+1]) # Transition probabilities.
        T[:,0] = 0.5 # Initialize first trial transition probabilities.
        Q_net = np.zeros(2)

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):

            # Generate trial events.
            c = choose(softmax(Q_net, 1.)) 
            s, o = task.trial(c)

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            # Update action values and transition probabilities.

            n = 1 - c  # Action not chosen at first step.
            r = 1 - s  # State not reached at second step.

            Q[n,i+1] = Q[n,i] 
            V[r,i+1] = V[r,i]
            T[n,i+1] = T[n,i]

            Q[c,i+1] = (1.-alpQ)*Q[c,i] + alpQ*((1.-lbd)*V[s,i] + lbd*o) # First step TD update.
            V[s,i+1] = (1.-alpQ)*V[s,i] + alpQ*o  # Second step TD update.

            T[c,i+1] = (1.-alpT)*T[c,i] + alpT*s  # Transition prob. update.

            # Evaluate net action values.

            Q_net = G_td*Q[:,i+1] + G_mb*(T[:,i+1]*V[1,i+1] + (1.-T[:,i+1])*V[0,i+1]) # Mixture of model based and model free values.
            Q_net[1] += bs
            Q_net[c] += 0.5*ck

        return choices, second_steps, outcomes