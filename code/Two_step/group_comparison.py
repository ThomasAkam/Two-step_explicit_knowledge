import numpy as np
from random import shuffle
import pylab as plt
import pickle
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, linregress
from pingouin import rm_anova
import statsmodels.formula.api as smf
import statsmodels.api as sm
from collections import OrderedDict
from . import model_plotting as mp
from . import plotting as pl
from . import model_fitting as mf
from . import parallel_processing as pp

# -------------------------------------------------------------------------------------
# Group comparison plots.
# -------------------------------------------------------------------------------------

def model_fit_comparison(sessions_A, sessions_B, agent, fig_no=1, title=None, ebars='pm95'):
    ''' Fit the two groups of sessions with the specified agent and plot the results on the same axis.'''
    eval_BIC = ebars == 'pm95'
    fit_A = mf.fit_population(sessions_A, agent, eval_BIC=eval_BIC)
    fit_B = mf.fit_population(sessions_B, agent, eval_BIC=eval_BIC)
    model_fit_comp_plot(fit_A, fit_B, fig_no=fig_no, sub_medians=True, ebars=ebars)
    if title:plt.suptitle(title)

def model_fit_comp_plot(fit_1, fit_2, fig_no=1, title=None, clf=True, sub_medians=True, ebars='pm95'):
    'Compare two different model fits.'
    mp.model_fit_plot(fit_1, fig_no, col='b', clf=clf , x_offset=-0.11, sub_medians=sub_medians, ebars=ebars)
    mp.model_fit_plot(fit_2, fig_no, col='r', clf=False, x_offset= 0.11, title=title, 
                      sub_medians=sub_medians, ebars=ebars)

def second_step_RT_comparison(sessions_A, sessions_B, fig_no=1, return_df=False):
    # Evaluate RTs for each subject, session and condition.
    print('sessions_A reaction times:')
    common_RTs_A, rare_RTs_A = pl.second_step_reaction_times(sessions_A, fig_no=False, col='b', return_med=True)
    print('sessions_B reaction times:')
    common_RTs_B, rare_RTs_B = pl.second_step_reaction_times(sessions_B, fig_no=False , col='r', return_med=True)
    # Make dataframe
    subject_IDs_A = [s.subject_ID for s in sessions_A]
    subject_IDs_B = [s.subject_ID for s in sessions_B]
    ns = len(subject_IDs_A)
    df = pd.DataFrame({
        'sID':subject_IDs_A*2+subject_IDs_B*2,
        'RT' : np.hstack([common_RTs_A, rare_RTs_A, common_RTs_B, rare_RTs_B]),
        'CR': ['com.']*ns + ['rare']*ns + ['com.']*ns + ['rare']*ns, 
        'AB': ['A']*ns*2 + ['B']*ns*2})
    if return_df: return df
    # Repeated measures Anova.
    aov = rm_anova(df, 'RT', within=['CR', 'AB'], subject='sID')
    print(aov.loc[:,['Source', 'ddof1','ddof2','F','p-unc','np2']])
    ttest = ttest_rel(common_RTs_A, common_RTs_B)
    print('Common A vs B t-test P value:{:.2e}, t:{:.2f}, df:{}'.format(
        ttest.pvalue, ttest.statistic, len(common_RTs_A)-1))
    ttest = ttest_rel(rare_RTs_A, rare_RTs_B)
    print('Rare   A vs B t-test P value:{:.2e}, t:{:.2f}, df:{}'.format(
        ttest.pvalue, ttest.statistic, len(common_RTs_A)-1))
    # Median RT bar plot.
    means = df.groupby(['CR','AB']).mean()['RT']
    sems  = df.groupby(['AB','CR']).sem()
    plt.figure(fig_no, figsize=[2.5,2.3], clear=True)
    means = df.groupby(['CR','AB']).mean()['RT']
    sems = df.groupby(['CR','AB']).sem()['RT']
    ax = sns.barplot(x='CR', y='RT', hue='AB',  data=df, ci=None, zorder=0)
    sns.swarmplot(x='CR', y='RT', hue='AB', dodge=True, data=df, color='k', size=2, zorder=0.5)
    ax.legend_.remove()
    plt.errorbar(x=[-0.2,0.2,0.8,1.2], y=means, yerr=sems, ecolor='r', ls='none')
    plt.ylabel('Reaction time (ms)')
    plt.tight_layout()
    return df

# -------------------------------------------------------------------
# Debriefing effect correlations.
# -------------------------------------------------------------------

def debriefing_effect_correlations(sessions_A, sessions_B, 
        trial_range_E=[10,20], trial_range_L=[30,40], fig_no=1):
    '''Correlates across subjects the change in the use of model-based
    RL over debreifing with the change in eligibility trace parameter. 
    Corelates the change in pereveration over debriefing (assessed with
    RL model) with the post debreifing change in perseveration from 
    early to late in blocks (assessed with logistic regression model).
    '''
    assert ([s.subject_ID for s in sessions_A] ==
            [s.subject_ID for s in sessions_B]
           ), 'Sessions A & B must have same subjects in same order'
    from .logistic_regression import config_log_reg
    from .RL_agents import MF_MBi
    RL_model  = MF_MBi(['bs','ck'])
    LR_model_E = config_log_reg(trial_select={'selection_type': 'rng',
                    'select_n': trial_range_E, 'block_type':'non_neutral'})
    LR_model_L = config_log_reg(trial_select={'selection_type': 'rng',
                    'select_n': trial_range_L, 'block_type':'non_neutral'})
    fits = {
    'RL_A'  : mf.fit_population(sessions_A, RL_model),
    'RL_B'  : mf.fit_population(sessions_B, RL_model),
    'LR_BE' : mf.fit_population(sessions_B, LR_model_E),
    'LR_BL' : mf.fit_population(sessions_B, LR_model_L)}
    sub_fits = {k: np.vstack([sf['params_T'] for sf in f['session_fits']])
                for k,f in fits.items()}
    RL_AB_diffs = {RL_model.param_names[i]: sub_fits['RL_B'][:,i] - 
        sub_fits['RL_A'][:,i] for i in range(RL_model.n_params)}
    LR_EL_diffs = {LR_model_E.param_names[i]: sub_fits['LR_BL'][:,i] - 
        sub_fits['LR_BE'][:,i] for i in range(LR_model_E.n_params)}
    plt.figure(fig_no, figsize=[6,3]).clf()
    # Correlate change over debreifing in G_mb with change in 'lbd'
    plt.subplot(1,2,1)
    sns.regplot(x=RL_AB_diffs['G_mb'], y=RL_AB_diffs['lbd'])
    plt.xlabel('Debriefing change in G_mb')
    plt.ylabel('Debriefing change in lambda')
    res = linregress(RL_AB_diffs['G_mb'], RL_AB_diffs['lbd'])
    print(res)
    print('Eligibility trace regression; slope: {:.3f} r: {:.3f} P value: {:.4f}'.format(
        res.slope, res.rvalue, res.pvalue))
    # Correlate debriefing effect on perseveration with post debreifing early vs late in block perseveration.
    plt.subplot(1,2,2)
    sns.regplot(RL_AB_diffs['ck'], LR_EL_diffs['choice'])
    plt.xlabel('Debriefing change in prsv')
    plt.ylabel('Within block change in prsv')
    res = linregress(RL_AB_diffs['ck'], LR_EL_diffs['choice'])
    print('Perseveration regression    ; slope: {:.3f} r: {:.3f} P value: {:.4f}'.format(
        res.slope, res.rvalue, res.pvalue))
    plt.tight_layout()

# -------------------------------------------------------------------
# Choice and motor prediction correlations.
# -------------------------------------------------------------------

def MB_motor_prediction_correlations(sessions, RL_agent, LR_agent):
    '''Plot the correlation between measures of subjects use of model-
    based RL and measures of their ability to predict upcoming motor
    actions at the second step.'''
    # Evalutate Transition x outcome predictor loadings for each session.
    LR_fit = mf.fit_population(sessions, LR_agent)
    TxO = [sf['params_T'][LR_agent.param_names.index('trCR_x_out')]
             for sf in LR_fit['session_fits']]
    # Evaluate RL model model-based weights.
    RL_fit = mf.fit_population(sessions, RL_agent)
    MB = [sf['params_T'][RL_agent.param_names.index('G_mb')]
             for sf in RL_fit['session_fits']]
    # Evaluate rare-common reaction time differences.
    RTs = pl.second_step_reaction_times(sessions, return_med=True, fig_no=False)
    dRT = np.array(RTs[1]) - np.array(RTs[0])
    # Evaluate rare-common invalid section step response rate difference. 
    IRs = pl.invalid_press_analysis(sessions, return_rates=True)
    dIR = IRs['s_r'] - IRs['s_c']
    # Plot correlations.
    df = pd.DataFrame({'MB':MB,'TxO':TxO,'dRT':dRT,'dIR':dIR})
    pairplot = sns.pairplot(df, kind="reg", plot_kws={'marker': '.'}, height=2)
    pairplot.axes[0,1].set_ylim(min(MB)-0.1,   max(MB) +0.2*(max(MB) -min(MB)))  # Set MB y axis range.
    pairplot.axes[1,0].set_ylim(min(TxO)-0.1,  max(TxO)+0.2*(max(TxO)-min(TxO))) # Set TxO y axis range.
    pairplot.axes[2,0].set_ylim(min(dRT)-20 ,  max(dRT)+0.2*(max(dRT)-min(dRT))) # Set dRT y axis range.
    pairplot.axes[3,0].set_ylim(min(dIR)-0.01, max(dIR)+0.2*(max(dIR)-min(dIR))) # Set dRT y axis range.
    pairplot.axes[1,0].set_xlim(min(MB)-0.1, max(MB)+0.1)     # Set MB y axis range.
    pairplot.axes[0,1].set_xlim(min(TxO)-0.1, max(TxO)+0.1)   # Set TxO y axis range.
    pairplot.axes[0,2].set_xlim(min(dRT)-20, max(dRT)+20)     # Set dRT y axis range.
    pairplot.axes[0,3].set_xlim(min(dIR)-0.01, max(dIR)+0.01) # Set dRT y axis range.
    # Print regression stats on plot axes.
    for Xind in range(4):
       for Yind in range(4):
            if Xind != Yind:
                fit = sm.OLS(df.iloc[:,Yind], sm.add_constant(df.iloc[:,Xind])).fit()
                printstr = f'R2={fit.rsquared :.2f} P={fit.pvalues[1] :.3f}'
                ax = pairplot.axes[Yind,Xind]
                ax.text(0.05, 0.9, printstr, transform=ax.transAxes)

# -------------------------------------------------------------------------------------
# Model fit permutation test.
# -------------------------------------------------------------------------------------

def model_fit_test(sessions_A, sessions_B, agent,  perm_type, n_perms=1000,
                   n_true_fit=5, title=None, file_name=None):

    '''Permutation test for significant differences in model fits between two groups of 
    sessions.  Outline of procedure:
    1. Perform model fitting seperately on both groups of sessions.
    2. Evaluate difference of means between fits for each parameter.
    3. Generate ensemble of resampled datasets in which sessions are randomly allocated
    to A or B.
    4. Perform model fitting and evalute differences for each resampled dataset to
    get a distribution of the differences under the null hypothesis that there is
    no difference between groups.
    5. Compare the true differences with the null distribution to get a P value.'''

    mf._precalculate_fits(sessions_A + sessions_B, agent) # Store first round fits on sessions.

    print('Fitting original dataset.')
    fit_test_data = {'param_names': agent.param_names[:],
                     'true_fits'  : pp.map(_fit_dataset,
                                      [(sessions_A, sessions_B, agent)]*n_true_fit)}

    perm_datasets = [_permuted_dataset(sessions_A, sessions_B, perm_type) + [agent] 
                     for _ in range(n_perms)]

    fit_test_data['perm_fits'] = []

    for i, perm_fit in enumerate(pp.imap(_fit_dataset, perm_datasets, ordered=False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9:
            _model_fit_p_values(fit_test_data, file_name)
            if file_name:
                with open(file_name+ '_test_data.pkl','wb') as data_file:
                    pickle.dump(fit_test_data, data_file)

    for session in sessions_A + sessions_B: del(session.fit) # Clear precalcuated fits.
    
def _fit_dataset(fit_data):
    # Evaluate and store fits for one dataset consisting of two sets of sessions,
    # along with differences between each parameter value.
    sessions_A, sessions_B, agent = fit_data   
    session_fits_A = [session.fit for session in sessions_A]
    session_fits_B = [session.fit for session in sessions_B] 
    fit_A = mf.fit_population(sessions_A, agent, session_fits=session_fits_A, verbose=False)
    fit_B = mf.fit_population(sessions_B, agent, session_fits=session_fits_B, verbose=False)
    if agent.type == 'RL': # Transform means from unconstrained to constrained space. 
        means_A = mf._trans_UT(fit_A['pop_dists']['means'], agent.param_ranges)
        means_B = mf._trans_UT(fit_B['pop_dists']['means'], agent.param_ranges)
    else:
        means_A = fit_A['pop_dists']['means']
        means_B = fit_B['pop_dists']['means']
    return {'means_A': means_A,
            'means_B': means_B,
            'differences': means_B-means_A}

def _model_fit_p_values(fit_test_data, file_name=None):
    '''Evaluate P values from differences between true and permuted datasets'''
    true_differences = np.median([f['differences'] for f in fit_test_data['true_fits']], axis=0)
    perm_differences = np.array([f['differences'] for f in fit_test_data['perm_fits']])
    diff_ranks = np.mean(perm_differences > true_differences, 0)
    p_values = 2*np.minimum(diff_ranks,1-diff_ranks)
    perm_diff_95ci = np.stack([np.percentile(perm_differences, 2.5, axis=0),
                               np.percentile(perm_differences, 97.5, axis=0)])
    n_perms = len(fit_test_data['perm_fits'])
    P_value_dict = OrderedDict([(pn,pv) for pn, pv in
                                zip(fit_test_data['param_names'], p_values)])
    diff_rank_dict = OrderedDict([(pn,dr) for pn, dr in
                                zip(fit_test_data['param_names'], diff_ranks)])
    fit_test_data.update({'true_differences': true_differences,
                          'perm_differences': perm_differences,
                          'p_values'        : P_value_dict,
                          'diff_ranks'      : diff_rank_dict,
                          'perm_diff_95ci'  : perm_diff_95ci,
                          'n_perms'         : n_perms})
    _print_test_results(fit_test_data, n_perms, file_name)

def _print_test_results(fit_test_data, n_perms=None, file_name=''):
    '''Print summary of permutation test showing for each parameter the true
    differences between groups, the 95% confidence interval of the difference
    under the null hypothesis, and the P values.'''
    if file_name: 
        _print_test_results(fit_test_data, n_perms, '') # Print to standard out then print to file.
    file = open(file_name + '.txt', 'w') if file_name else None
    if file_name: print(file_name)
    print('Permutation test data' + (' ({} permutations):'.format(n_perms) if n_perms else ':'), file=file)
    name_len = max([len(name) for name in fit_test_data['param_names'] + ['Parameter']])
    print('Parameter'.ljust(name_len) + '| Data difference |  Permuted difference 95%CI | P value', file=file)
    for i, pn in enumerate(fit_test_data['param_names']):
        td = fit_test_data['true_differences'][i]
        ci = fit_test_data['perm_diff_95ci'][:,i]
        pv = fit_test_data['p_values' ][pn]
        stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        print(pn.ljust(name_len) + '| ' + f'{td :8.3f}'.center(15) + ' | ' + 
              f'({ci[0]:8.3f},{ci[1]:8.3f})'.center(26) + f' | {pv:5.4f} ' + stars, file=file) 

# -------------------------------------------------------------------------------------
# Interaction permutation test.
# -------------------------------------------------------------------------------------

# Model fit interaction test ----------------------------------------------------------

def model_fit_interaction_test(sessions_XA, sessions_XB, sessions_YA, sessions_YB, agent,
                               n_perms=1000, n_true_fit=5, file_name=None):

    '''Permutation test for whether the effect of condition A vs B is different in groups X vs Y.'''

    _interaction_check_groups(sessions_XA, sessions_XB, sessions_YA, sessions_YB)

    # Store first round fits on sessions.
    int_fit_data = sessions_XA, sessions_XB, sessions_YA, sessions_YB, agent 
    mf._precalculate_fits(sessions_XA + sessions_XB + sessions_YA + sessions_YB, agent) 

    print('Fitting original dataset.')
    fit_test_data = {'param_names': agent.param_names[:],
                     'true_fits' :pp.map(_effect_size_difference, [int_fit_data]*n_true_fit)}
    
    perm_datasets = [_interaction_permuted_dataset(sessions_XA, sessions_XB, sessions_YA, sessions_YB)
                     + [agent] for _ in range(n_perms)]

    fit_test_data['perm_fits'] = []

    for i, perm_fit in enumerate(pp.imap(_effect_size_difference, perm_datasets, ordered=False)):
        fit_test_data['perm_fits'].append(perm_fit)
        print('Fitted permuted dataset {} of {}'.format(i+1, n_perms))
        if i > 0 and i%10 == 9:
            _model_fit_p_values(fit_test_data, file_name)
            if file_name:
                with open(file_name+ '_test_data.pkl','wb') as data_file:
                    pickle.dump(fit_test_data, data_file)
                    
    for session in sessions_XA + sessions_XB + sessions_YA + sessions_YB:
        del(session.fit) # Clear precalcuated fits.


def _effect_size_difference(int_fit_data):
    '''Evaluate the difference between the effect of condition  A vs B in group X vs Y.'''
    sessions_XA, sessions_XB, sessions_YA, sessions_YB, agent = int_fit_data    
    
    fits_X = _fit_dataset((sessions_XA, sessions_XB, agent))
    fits_Y = _fit_dataset((sessions_YA, sessions_YB, agent))
    
    differences = fits_Y['differences'] - fits_X['differences']
    
    return {'fits_X': 'fits_X',
            'fits_Y': 'fits_Y',
            'differences': differences}
            
# Interaction test shared functions. -----------------------------------------------------

def _interaction_permuted_dataset(sessions_XA, sessions_XB, sessions_YA, sessions_YB):
    '''Permute subjects between groups X and Y then create permuted sessions which respect
    the permuted group assignments X and Y and the true conditions A vs B.'''

    all_sessions_A = sessions_XA + sessions_YA
    all_sessions_B = sessions_XB + sessions_YB
   
    #Get subject IDs
    subjects_X = list(set([s.subject_ID for s in sessions_XA]))
    subjects_Y = list(set([s.subject_ID for s in sessions_YA])) 
    n_subject_X   = len(subjects_X)                        
    
    #Permute subject across groups
    all_subject_IDs = subjects_X + subjects_Y
    shuffle(all_subject_IDs)
    perm_subjects_X = all_subject_IDs[:n_subject_X]  #respect group size
    perm_subjects_Y = all_subject_IDs[n_subject_X:]

    #Create session list respecting permuted groups X vs Y and true conditions A vs B.
    perm_ses_XA = [s for s in all_sessions_A if s.subject_ID in perm_subjects_X]
    perm_ses_XB = [s for s in all_sessions_B if s.subject_ID in perm_subjects_X]
    perm_ses_YA = [s for s in all_sessions_A if s.subject_ID in perm_subjects_Y]
    perm_ses_YB = [s for s in all_sessions_B if s.subject_ID in perm_subjects_Y]
    
    return [perm_ses_XA, perm_ses_XB, perm_ses_YA, perm_ses_YB]
    
def _interaction_check_groups(sessions_XA, sessions_XB, sessions_YA, sessions_YB):
    assert set([s.subject_ID for s in sessions_XA]) == set([s.subject_ID for s in sessions_XB]), \
        'Sessions XA and XB must comprise the same group of subjects.'
    assert set([s.subject_ID for s in sessions_YA]) == set([s.subject_ID for s in sessions_YB]), \
        'Sessions YA and YB must comprise the same group of subjects.'

# Reaction time interaction test --------------------------------------------------------

def second_step_RT_interaction_test(sessions_XA, sessions_XB, 
                                    sessions_YA, sessions_YB, file_name=None):
    '''Mixed linear model for whether the effect on second step RT of
    condition A vs B is different in groups X vs Y. Note - now using
    Mixed ANOVA calcualted in R for this using saved dataframe, as 
    statsmodels mixedlm does not give effect sizes.'''
    dfX = second_step_RT_comparison(sessions_XA, sessions_XB, return_df=True)
    dfY = second_step_RT_comparison(sessions_YA, sessions_YB, return_df=True)
    dfX['group'] = 'X'
    dfY['group'] = 'Y'
    df = pd.concat([dfX, dfY])
    md = smf.mixedlm('RT ~ CR * AB * group', df, groups=df['sID'])
    print(md.fit().summary())
    if file_name:
        df.to_csv(file_name + '.csv')

#---------------------------------------------------------------------------------------------------
#  Permuted dataset generation.
#---------------------------------------------------------------------------------------------------

def _permuted_dataset(sessions_A, sessions_B, perm_type):
    ''' Generate permuted datasets by randomising assignment of sessions between groups A and B.
    perm_type argument controls how permutations are implemented:
    'within_subject' - Permute sessions within subject such that each permuted group has the same
                     number of session from each subject as the true datasets.
    'cross_subject' - All sessions from a given subject are assigned to one or other of the permuted datasets.
    'ignore_subject' - The identity of the subject who generated each session is ignored in the permutation.
    'within_group' - Permute subjects within groups that are subsets of all subjects.  
                     Animal assignment to groups is specified by groups argument which should be 
                     a list of lists of animals in each grouplt.
    '''
    assert perm_type in ('within_subject', 'cross_subject'), 'Invalid permutation type.'
    all_sessions = sessions_A + sessions_B
    all_subjects = list(set([s.subject_ID for s in all_sessions]))

    if perm_type == 'cross_subject':  # Permute subjects across groups (used for cross subject tests.)
        n_subj_A     = len(set([s.subject_ID for s in sessions_A]))        
        shuffle(all_subjects)   
        perm_ses_A = [s for s in all_sessions if s.subject_ID in all_subjects[:n_subj_A]]
        perm_ses_B = [s for s in all_sessions if s.subject_ID in all_subjects[n_subj_A:]]
    
    elif perm_type == 'within_subject': # Permute sessions keeping number from each subject in each group constant.
        perm_ses_A = []
        perm_ses_B = []
        for subject in all_subjects:
            subject_sessions_A = [s for s in sessions_A if s.subject_ID == subject]
            subject_sessions_B = [s for s in sessions_B if s.subject_ID == subject]
            all_subject_sessions = subject_sessions_A + subject_sessions_B
            shuffle(all_subject_sessions)
            perm_ses_A += all_subject_sessions[:len(subject_sessions_A)]
            perm_ses_B += all_subject_sessions[len(subject_sessions_A):]

    return [perm_ses_A, perm_ses_B]
