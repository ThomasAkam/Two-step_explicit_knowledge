import numpy as np
import pylab as plt
from functools import partial
from collections import Counter
from scipy.stats import chi2
from . import utility as ut
from . import model_fitting as mf
from . import parallel_processing as pp

def BIC_model_comparison(sessions, agents, n_draws=5000, n_repeats=1, fig_no=1,
                         file_name=None, log_Y=False):
    ''' Compare goodness of different fits using integrated BIC.'''    
    if n_repeats > 1: 
        fit_func = partial(mf.repeated_fit_population, sessions, n_draws=n_draws, n_repeats=n_repeats)
        fiterator = map(fit_func, agents) # Save parallel processessing for repeated fits of same agent.
    else:
        fit_func = partial(mf.fit_population, sessions, eval_BIC={'n_draws':n_draws})
        fiterator = pp.imap(fit_func, agents, ordered=False) # Use parallel processing for seperate agents.
    population_fits = []
    for i,fit in enumerate(fiterator):
        print('Fit {} of {}, agent: '.format(i+1, len(agents)) + fit['agent_name'])
        population_fits.append(fit)
        if file_name: ut.save_item(population_fits, file_name)
    BIC_comparison_plot(population_fits, fig_no, log_Y)
    return population_fits

def BIC_comparison_plot(population_fits, fig_no=1, log_Y=False, plot_rep_fits=False):
    '''Plot the results of a BIC model comparison'''
    sorted_fits = sorted(population_fits, key = lambda fit: fit['iBIC']['score'])
    print('BIC_scores:')
    for fit in sorted_fits:
        s =   '{:.3f} : '.format(fit['iBIC']['best_prob']) if 'best_prob' in fit['iBIC'].keys() else ''
        print('{:.0f} : '.format(round(fit['iBIC']['score'])) + s + fit['agent_name'])
    print('The best fitting model is: ' + sorted_fits[0]['agent_name'])
    if fig_no:
        BIC_scores = np.array([fit['iBIC']['score'] for fit in sorted_fits])
        BIC_deltas = BIC_scores - BIC_scores[0]
        agent_names = [fit['agent_name'] for fit in sorted_fits]
        x = np.flipud(np.arange(1,len(agent_names)+1))
        if 'BIC_95_conf' in fit['iBIC'].keys():
            ebars = np.array([np.abs(fit['iBIC']['BIC_95_conf'] - fit['iBIC']['score'])
                              for fit in sorted_fits]).T
        else: 
            ebars = -2*np.array([np.abs(fit['iBIC']['lik_95_conf'] - fit['iBIC']['int_lik'])
                                for fit in sorted_fits]).T
        plt.figure(fig_no, figsize=[3,3]).clf()
        plt.bar(x, BIC_deltas, color = 'k')
        plt.errorbar(x, BIC_deltas, ebars, color = 'r', linestyle = '', elinewidth = 2)
        if 'repeated_fits' in population_fits[0].keys() and plot_rep_fits: # Scatter plot repeated fits.
            for fit, xi in zip(sorted_fits, x):
                rep_fit_iBICs = np.array([f['iBIC']['score'] for f in fit['repeated_fits']])
                plt.scatter(xi+0.4+np.linspace(-0.2,0.2,len(rep_fit_iBICs)), rep_fit_iBICs - BIC_scores[0])
        plt.xticks(x, agent_names, rotation = -45, ha = 'left')
        plt.xlim(0.5,len(agent_names)+0.5)
        plt.ylim(0, BIC_deltas[-1]*1.2)
        plt.ylabel('∆ BIC')
        plt.figtext(0.13,0.92,'Best BIC score: {}'.format(int(BIC_scores[0])))
        plt.tight_layout()
        if log_Y:
            plt.gca().set_yscale('log')
            plt.ylim(10,plt.ylim()[1])

def store_fits_on_sessions(sessions, agents, use_prior=False, repeats=5):
    ''' Fit each agent to each session and store the fit in a dictionary on each session whose keys
    are the agent names.'''
    fit_func = partial(_fit_agents_to_session, agents=agents, use_prior=use_prior, repeats=repeats)
    for i, session_fits in enumerate(pp.imap(fit_func, sessions)):
        print('Fitting session {} of {}. '.format(i+1, len(sessions)))
        try:
            sessions[i].fits.update(session_fits)
        except AttributeError:
            sessions[i].fits = session_fits

def _fit_agents_to_session(session, agents, use_prior, repeats):
    return {agent.name + ('_p' if use_prior else ''):  mf.fit_session_con(session, agent, use_prior=use_prior, repeats=repeats)
            for agent in agents}


def two_agent_per_subject_comp(sessions,  agent_name_A='MF_MBi_bs_ck', agent_name_B='MF_bs_ck',
                               metric='loglik', fig_no=1, title=None, return_subs=False, 
                               return_diffs=False, ymax=None):
    '''Bar plot showing difference in fit quality between agent A and agent B for each subject, ranked by 
    fit quality difference.  Metric can be BIC score or likelihood.  IF metric is BIC score, subjects bars 
    are colour coded by which agent has higher BIC score.  If metric is likelihood, it is assumed that 
    agent B is a special case of agent A, and the bars are coloured according to which agent
    is best according to a likelihood ratio test.'''
    assert metric in ('BIC', 'loglik', 'AIC', 'logpostprob')
    if metric in ('BIC','AIC'):
        sign = -1     # Low values indicate good fits.
        threshold = 0 # Difference in metric above which to select agent A.
    elif metric in ('loglik', 'logpostprob'):
        sign = 1 # High values indicate good fits.
        ses_per_sub = list(Counter([s.subject_ID for s in sessions]).values())
        assert len(set(ses_per_sub)) == 1, \
            'Likelihood threshold only valid with same number of sessions per subject.'
        threshold =  chi2.isf(0.05, ses_per_sub[0]*(sessions[0].fits[agent_name_A]['n_params'] - 
                                                    sessions[0].fits[agent_name_B]['n_params']))/2
        print('Threshold: {}'.format(threshold))
    fit_quality_diffs = {}
    subject_IDs = list(set([s.subject_ID for s in sessions]))
    for sID in subject_IDs:
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        fit_quality_diffs[sID] = sign*sum([s.fits[agent_name_A][metric]-s.fits[agent_name_B][metric]
                                           for s in subject_sessions])
    if return_diffs: return fit_quality_diffs
    A_better_subs = sorted([sID for sID in subject_IDs if fit_quality_diffs[sID]> threshold])
    B_better_subs = sorted([sID for sID in subject_IDs if fit_quality_diffs[sID]<=threshold])
    if return_subs: return A_better_subs, B_better_subs   
    sorted_fit_quality_diffs = np.array(sorted(list(fit_quality_diffs.values())))
    plt.figure(fig_no, figsize=[3.3,3]).clf()
    bar_list = plt.bar(np.arange(sorted_fit_quality_diffs.shape[0]), sorted_fit_quality_diffs, width=1)
    for i, fit_quality_diff in enumerate(sorted_fit_quality_diffs):
        if fit_quality_diff > threshold: bar_list[i].set_facecolor('r')
    plt.plot([0,len(fit_quality_diffs)],[threshold, threshold],':k')
    plt.xlim(0,len(fit_quality_diffs))
    if metric == 'loglik': 
        plt.ylim(bottom=0)
        if sorted_fit_quality_diffs[0]<-1e-5:
            print('Warning: negative likelihood difference: {}'.format(sorted_fit_quality_diffs[0]))
    if ymax: plt.ylim(top=ymax)
    plt.text(0.05, 0.85, '{}/{}'.format(len(A_better_subs), len(subject_IDs)), transform=plt.gca().transAxes)
    plt.xlabel('Subject #')
    plt.ylabel('$\Delta$ {}\n{} vs {}'.format(metric, agent_name_A, agent_name_B))
    if title: plt.title(title)
    plt.tight_layout()
    print('{} of {} subjects ({:.1f}%) best desribed by more complex model {}'.format(
          len(A_better_subs), len(subject_IDs), 100*len(A_better_subs)/len(subject_IDs), agent_name_A)) 