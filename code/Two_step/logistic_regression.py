import numpy as np
from functools import partial
from . import utility as ut
from . import model_fitting as mf 
from . import model_plotting as mp
from . import parallel_processing as pp
     
# -------------------------------------------------------------------------------------
# Configurable logistic regression Model.
# -------------------------------------------------------------------------------------

class config_log_reg():

    '''
    Configurable logistic regression agent. Arguments:

    predictors - The basic set of predictors used is specified with predictors argument.  

    lags        - By default each predictor is only used at a lag of -1 (i.e. one trial predicting the next).
                 The lags argument is used to specify the use of additional lags for specific predictors:
                 e.g. lags = {'outcome': 3, 'choice':2} specifies that the outcomes on the previous 3 trials
                 should be used as predictors, while the choices on the previous 2 trials should be used.  If an
                 interger is provided as the lags argument all predictors are given this number of lags.
    '''

    def __init__(self, predictors=['correct','choice','outcome', 'trans_CR', 'trCR_x_out'],
                 lags={}, trial_select='default'):

        self.name = 'LR_model'

        self.base_predictors = predictors # predictor names ignoring lags.

        if type(lags) == int:
            lags = {p:lags for p in predictors}

        self.predictors = [] # predictor names including lags.
        for predictor in self.base_predictors:
            if predictor in list(lags.keys()):
                for i in range(lags[predictor]):
                    self.predictors.append(predictor + '-' + str(i + 1)) # Lag is indicated by value after '-' in name.
            else:
                self.predictors.append(predictor) # If no lag specified, defaults to 1.

        self.n_predictors = len(self.predictors)

        if trial_select == 'default':
            self.trial_select = {'selection_type': 'xtr',
                                 'select_n'      : 20,
                                 'block_type'    :'all'}      
        else:
            self.trial_select = trial_select

        self.n_params = 1 + len(self.predictors)
        self.param_ranges = ('all_unc', self.n_params)
        self.param_names  = ['bias'] + self.predictors
        self.calculates_gradient = True
        self.type = 'log_reg'


    def session_likelihood(self, session, params_T, eval_grad=False):

        bias = params_T[0]
        weights = params_T[1:]

        choices = session.trial_data['choices']

        if not hasattr(session,'predictors'):
            predictors = self._get_session_predictors(session) # Get array of predictors
        else:
            predictors = session.predictors

        assert predictors.shape[0] == session.n_trials, 'predictor array does not match number of trials.'
        assert predictors.shape[1] == len(weights), 'predictor array does not match number of weights.'

        if self.trial_select: # Only use subset of trials.
            if self.trial_select == 'session_specific':
                selected_trials = session.select_trials(**session.trial_select)
            else: 
                selected_trials = session.select_trials(**self.trial_select)
            choices = choices[selected_trials]
            predictors = predictors[selected_trials,:]

        # Evaluate session log likelihood.

        Q = np.dot(predictors,weights) + bias
        Q[Q < -ut.log_max_float] = -ut.log_max_float # Protect aganst overflow in exp.
        P = 1./(1. + np.exp(-Q))  # Probability of making choice 1
        Pc = 1 - P - choices + 2. * choices * P  # Probability of chosen action.

        session_log_likelihood = sum(ut.log_safe(Pc)) 

        # Evaluate session log likelihood gradient.

        if eval_grad:
            dLdQ  = - 1 + 2 * choices + Pc - 2 * choices * Pc
            dLdB = sum(dLdQ) # Likelihood gradient w.r.t. bias paramter.
            dLdW = sum(np.tile(dLdQ,(len(weights),1)).T * predictors, 0) # Likelihood gradient w.r.t weights.
            session_log_likelihood_gradient = np.append(dLdB,dLdW)
            return (session_log_likelihood, session_log_likelihood_gradient)
        else:
            return session_log_likelihood


    def _get_session_predictors(self, session):
        'Calculate and return values of predictor variables for all trials in session.'

        # Evaluate base (non-lagged) predictors from session events.

        choices, transitions_AB, second_steps, outcomes = session.unpack_trial_data(dtype = bool)
        trans_state = session.blocks['trial_trans_state']    # Trial by trial state of the tranistion matrix (A vs B)
        transitions_CR = transitions_AB == trans_state
        transition_CR_x_outcome = transitions_CR == outcomes 
        correct = -0.5*(session.blocks['trial_rew_state']-1)* \
                       (2*session.blocks['trial_trans_state']-1) 
        transitions_CR = transitions_AB == trans_state
        transition_CR_x_outcome = transitions_CR == outcomes 

        bp_values = {} 

        for p in self.base_predictors:

            if p == 'correct':  # 0.5, 0, -0.5 for high poke being correct, neutral, incorrect option.
                bp_values[p] =  correct
      
            elif p == 'side': # 0.5, -0.5 for left, right side reached at second step. 
                bp_values[p] = second_steps - 0.5

            elif p ==  'choice': # 0.5, - 0.5 for choices high, low.
                bp_values[p] = choices - 0.5
                    
            elif p ==  'outcome': # 0.5 , -0.5 for  rewarded , not rewarded.
                bp_values[p] = (outcomes == choices) - 0.5

            elif p ==  'trans_CR': # 0.5, -0.5 for common, rare transitions.  
                bp_values[p] = ((transitions_CR) == choices)  - 0.5               

            elif p == 'trCR_x_out': # 0.5, -0.5 for common, rare transitions inverted by trial outcome.
                bp_values[p] = (transition_CR_x_outcome  == choices) - 0.5

            elif p == 'rew_com':  # Rewarded common transition predicts repeating choice.
                bp_values[p] = ( outcomes &  transitions_CR) * (choices - 0.5)

            elif p == 'rew_rare':  # Rewarded rare transition predicts repeating choice.
                bp_values[p] = ( outcomes & ~transitions_CR) * (choices - 0.5)   

            elif p == 'non_com':  # Non-rewarded common transition predicts repeating choice.
                bp_values[p] = (~outcomes &  transitions_CR) * (choices - 0.5)

            elif p == 'non_rare':  # Non-Rewarded rare transition predicts repeating choice.
                bp_values[p] = (~outcomes & ~transitions_CR) * (choices - 0.5)

        # Generate lagged predictors from base predictors.

        predictors = np.zeros([session.n_trials, self.n_predictors])

        for i,p in enumerate(self.predictors):  
            if '-' in p: # Get lag from predictor name.
                lag = int(p.split('-')[1]) 
                bp_name = p.split('-')[0]
            else:        # Use default lag.
                lag = 1
                bp_name = p
            predictors[lag:, i] = bp_values[bp_name][:-lag]

        return predictors

# -------------------------------------------------------------------------------------
# Bootstrap significance testing.
# -------------------------------------------------------------------------------------

def predictor_significance_test(sessions, agent, n_perms=1000, file_name=None):
    '''Test whether logistic regression predictor loadings are significantly
    different from zero by bootstrap resampling subjects.'''

    mf._precalculate_fits(sessions, agent) # Store first round fits on sessions.
    
    true_means = mf.fit_population(sessions, agent, session_fits=[session.fit for session in sessions])['pop_dists']['means']

    permute_and_fit = partial(_permute_and_fit, agent=agent)

    bootstrap_fits = []
    for i, bs_fit in enumerate(pp.imap(permute_and_fit, [sessions]*n_perms, ordered=False)):
        bootstrap_fits.append(bs_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9:
            stats_dict = _eval_stats(bootstrap_fits, true_means, agent)
            _print_stats(stats_dict, n_perms=i+1, file_name=file_name)

    for session in sessions: del(session.fit) # Clear precalcuated fits.

    return bootstrap_fits

def _permute_and_fit(sessions, agent):
    resampled_sessions = ut.resample_subjects(sessions)
    rs_fits = [session.fit for session in resampled_sessions]   
    return mf.fit_population(resampled_sessions, agent, session_fits=rs_fits, verbose=False)

def _eval_stats(bootstrap_fits, true_means, agent):
    pop_means = np.array([bf['pop_dists']['means'] for bf in bootstrap_fits])
    P_values = np.min((np.mean(pop_means > 0, 0), np.mean(pop_means < 0, 0)),0)*2.
    CI95 = np.stack([np.percentile(pop_means, 2.5, axis=0),
                     np.percentile(pop_means, 97.5, axis=0)])
    return {'param_names': agent.param_names,
            'true_means' : true_means,
            'P values'   : P_values,
            'CI95'       : CI95}

def _print_stats(stats_dict, n_perms=None, file_name=None):
    if file_name: 
        _print_stats(stats_dict, n_perms) # Print to standard out then print to file.
    file = open(file_name + '.txt', 'w') if file_name else None
    print('P values' + (' ({} permutations):'.format(n_perms) if n_perms else ':'), file=file)
    name_len = max([len(name) for name in stats_dict.keys()])
    print('Parameter'.ljust(name_len) + '| Data value |  bootstrap 95%CI | P value', file=file)
    for i, pn in enumerate(stats_dict['param_names']):
        tm = stats_dict['true_means'][i]
        ci = stats_dict['CI95'][:,i]
        pv = stats_dict['P values'][i]
        stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        print(pn.ljust(name_len) + '|' + f'{tm :6.3f}'.center(11) + ' | ' + 
              f'({ci[0]:6.3f},{ci[1]:6.3f})'.center(16) + f' | {pv:5.4f} ' + stars, file=file) 
    if file_name: file.close()