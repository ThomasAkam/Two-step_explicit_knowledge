import numpy as np
import pylab as plt
from random import choice, random
from copy import deepcopy
from . import model_fitting as mf
from .human_session import human_session

#------------------------------------------------------------------------------------
# Two-step task.
#------------------------------------------------------------------------------------

class Two_step_task:
    '''Two step task model.'''
    def __init__(self):
        # Parameters
        self.norm_prob = 0.8 # Probability of normal transition.

        self.reward_probs = np.array([[0.2, 0.8],  # Reward probabilities in each reward block type.
                                      [0.4, 0.4],
                                      [0.8, 0.2]])
        self.threshold = 0.75 
        self.tau = 8.  # Time constant of moving average.
        self.min_block_length = 40      # Minimum block length.
        self.post_threshold_delay = 20  # Number of trials after transition threshold reached before transtion occurs.
        self.mov_ave = _exp_mov_ave(tau = self.tau, init_value = 0.5)   # Moving average of agents choices.
        self.reset()

    def reset(self, n_trials=1000):
        self.reward_block = choice([0,2]) # 0 for left good, 2 for right good.
        self.old_reward_block = None
        self.block_trials = 0             # Number of trials into current block.
        self.cur_trial = 0                # Current trial number.
        self.threshold_crossed = False    # True if threshold % correct to trigger block transition has been reached.
        self.n_post_threhold = 0          # Current number of trials past criterion.
        self.trial_number = 1             # Current trial number.
        self.n_trials = n_trials          # Session length.
        self.mov_ave.reset()
        self.blocks = {'start_trials'      : [0],
                       'end_trials'        : [],
                       'reward_states'     : [self.reward_block], # 0 for left good, 1 for neutral, 2 for right good.
                       'transition_states' : [1]}  # Only used for version of task with changing reward probs.

    def trial(self, choice):

        self.block_trials += 1
        self.cur_trial += 1
        self.mov_ave.update(choice)
        second_step = int(choice == _with_prob(self.norm_prob))
        outcome = int(_with_prob(self.reward_probs[self.reward_block, second_step]))
        # Check for block transition.
        block_transition = False
        if self.reward_block == 1: # Neutral block
            if (self.block_trials > 40) & _with_prob(0.1):
                block_transition = True
        else: # Non neutral block.
            if not self.threshold_crossed: # Check if threshold has been crossed.
                if (self.reward_block == 0) & (self.mov_ave.ave > self.threshold):
                    self.threshold_crossed = True
                elif (self.reward_block == 2) & (self.mov_ave.ave < (1. -self.threshold)):
                    self.threshold_crossed = True
            else: # Threshold already crossed, check for transition.
                self.n_post_threhold +=1
                if (self.n_post_threhold >= self.post_threshold_delay) & \
                   (self.block_trials >= self.min_block_length):
                   block_transition = True
           
        if block_transition:
            self.block_trials = 0
            self.n_post_threhold = 0
            self.threshold_crossed = False

            if self.reward_block == 1: # End of neutral block.
                self.reward_block = 2 - self.blocks['reward_states'][-2] 
            else: # End of non-neutral block
                if _with_prob(0.5): 
                        self.reward_block = 1 # Transition to neutral block.
                else:
                    self.reward_block = 2 - self.reward_block # Invert reward probs.

            self.blocks['start_trials'].append(self.cur_trial)
            self.blocks['end_trials'].append(self.cur_trial)
            self.blocks['reward_states'].append(self.reward_block)
            self.blocks['transition_states'].append(1)

        if self.cur_trial >= self.n_trials: #End of session.
            self.blocks['end_trials'].append(self.cur_trial + 1)

            self.blocks['trial_trans_state'] = np.zeros(self.n_trials, dtype = bool) # Boolean array indication state of tranistion matrix for each trial.
            self.blocks['trial_rew_state']   = np.zeros(self.n_trials, dtype = int)

            for start_trial,end_trial, trans_state, reward_state in \
                    zip(self.blocks['start_trials'],self.blocks['end_trials'], \
                        self.blocks['transition_states'], self.blocks['reward_states']):
                self.blocks['trial_trans_state'][start_trial:end_trial] = trans_state   
                self.blocks['trial_rew_state'][start_trial:end_trial]  = reward_state   

        return (second_step, outcome)

class _exp_mov_ave:
    'Exponential moving average class.'
    def __init__(self, tau=None, init_value=0., alpha = None):
        if alpha is None: alpha = 1 - np.exp(-1/tau)
        self._alpha = alpha
        self._m = 1 - alpha
        self.init_value = init_value
        self.reset()

    def reset(self, init_value = None):
        if init_value:
            self.init_value = init_value
        self.ave = self.init_value

    def update(self, sample):
        self.ave = (self.ave*self._m) + (self._alpha*sample)


def _with_prob(prob):
    'return true / flase with specified probability .'
    return random() < prob

#------------------------------------------------------------------------------------
# Simulation.
#------------------------------------------------------------------------------------

class simulated_session(human_session):
    '''Stores agent parameters and simulated data, supports plotting as for experimental
    session class.
    '''
    def __init__(self, task, agent, params_T, n_trials=300, subject_ID=-1, number=1):
        '''Simulate session with current agent and task parameters.'''
        self.param_names = agent.param_names
        self.true_params_T = params_T
        self.subject_ID = subject_ID
        self.number = number
        try: # Not possible for e.g. unit range params_T with value 0 or 1.
            self.true_params_U = mf.transTU(params_T, agent.param_ranges)
        except Exception: 
            self.true_params_U = None
        self.n_trials = n_trials
        choices, second_steps, outcomes = agent.simulate(task, params_T, n_trials)
        
        self.trial_data = {'choices'      : choices,
                           'transitions'  : (choices == second_steps).astype(int),
                           'second_steps' : second_steps,
                           'outcomes'     : outcomes}

        if hasattr(task,'blocks'):
            self.blocks = deepcopy(task.blocks)
 
def simulate_sessions_from_fit(sessions, agent, n_trials=300):
    '''Fit agent to sessions then simulate a set of sessions using the 
    MAP parameters for each of the original session fits.'''
    fit = mf.fit_population(sessions, agent)
    task = Two_step_task()
    fit_params_T = np.array([sf['params_T'] for sf in fit['session_fits']])
    sim_sessions = [simulated_session(task, agent, params_T, n_trials, i)
                    for i, params_T in enumerate(fit_params_T)]
    return sim_sessions

#--------------------------------------------------------------------------------------
# Parameter recovery
#--------------------------------------------------------------------------------------

def parameter_recovery(sessions, agent, n_points=9, n_reps=10, 
                       n_trials=300, fig_no=1, filename=None,
                       rec_params=None, fix_params=None):
    '''Test how accuractely population mean paramter values can be 
    recovered from simualated data.  The provided sessions both
    determines the number of simulated sessions used for paramter
    recovery, and the values that paramters not currently being
    evalutater are set to.'''
    rec_params if rec_params else agent.param_names # Paramters to test recovery for.
    # Fit RL model to sessions.
    fit = mf.fit_population(sessions, agent)
    # Check recovery for each parameter when other parameters are drawn from fit.
    task = Two_step_task()
    test_values = _get_test_values(n_points)
    recovered_params = np.zeros([agent.n_params, n_points, n_reps]) 
    fit_n = 0   
    for i, (p_name, p_range) in enumerate(zip(agent.param_names, 
                                            agent.param_ranges)):
        if not p_name in rec_params:
            continue
        for v, p_value in enumerate(test_values[p_range]):
            for r in range(n_reps):
                fit_n += 1
                print(f'Fit {fit_n} of {len(rec_params)*n_points*n_reps}')
                # Create session paramters.
                ses_params_T = np.array([mf._sample_params_T(fit) for _ in range(len(sessions))])
                ses_params_T[:,i] = p_value
                if fix_params: # Set specified parameters to fixed values.
                    for fp in fix_params.keys():
                        ses_params_T[:,agent.param_names.index(fp)] = fix_params[fp]
                # Simulate sessions and fit.
                sim_sessions = [simulated_session(task, agent, params_T, n_trials, s, sessions[0].number)
                                for s, params_T in enumerate(ses_params_T)]
                sim_fit = mf.fit_population(sim_sessions, agent)
                means_T = mf._trans_UT(sim_fit['pop_dists']['means'], agent.param_ranges)
                recovered_params[i, v, r] = means_T[i]
    if filename:
        np.save(filename+'.npy', recovered_params)
    else:
        _parameter_recovery_plot(recovered_params, agent, fig_no)


def _parameter_recovery_plot(recovered_params, agent, fig_no=1):
    rec_means = np.mean(recovered_params, 2)
    rec_SDs = np.std(recovered_params, 2)
    test_values = _get_test_values(rec_means.shape[1])
    plt.figure(fig_no, figsize=[2*agent.n_params,2.5], clear=True)
    xlims = {'unit': [0,1],
             'pos' : [0,5],
             'unc' : [-2.5,2.5]}
    for i in range(agent.n_params):
        plt.subplot(1, agent.n_params, i+1)
        p_range = agent.param_ranges[i]
        plt.errorbar(test_values[p_range], rec_means[i,:], 
                     yerr=rec_SDs[i,:], marker='.')
        plt.plot(xlims[p_range],xlims[p_range],':k')
        plt.xlim(*xlims[p_range])
        plt.ylim(*xlims[p_range])
        plt.title(agent.param_names[i])
        if i == 0: 
            plt.ylabel('Recovered')
        elif i == np.floor(agent.n_params/2):
            plt.xlabel('True paramter value')
    plt.tight_layout()

def _get_test_values(n_points):
    return {'unit': np.linspace(0.1,0.9,n_points),
            'pos' : np.linspace(0.5,4.5,n_points),
            'unc' : np.linspace( -2, 2 ,n_points)}