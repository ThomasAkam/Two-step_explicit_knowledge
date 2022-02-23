'''Script with functions to generate figures and run permutation tests for clinical groups.'''

from Two_step import lr, gc, pp, he, rl, mc, pl

pp.enable_multiprocessing() # Use multiprocessing for model fitting.

#------------------------------------------------------------------------------
# Logistic regression and RL models.
#------------------------------------------------------------------------------

LR_agent = lr.config_log_reg()
MF_agent = rl.MF()
RL_agent = rl.MF_MBi()

#-----------------------------------------------------
# Import data
#-----------------------------------------------------

# The first time the script is imported the fits needed to identify subjects  
# who are MF and MB at session 3 are calculated which will take a while. These
# fits are then saved making subsequent imports much faster.

exp_hvs_NY    = he.experiment('healthy-volunteers_fixed-task_debrief_NY')
exp_hvs_LS    = he.experiment('healthy-volunteers_fixed-task_debrief_LS')
exp_hvs_LSnod = he.experiment('healthy-volunteers_fixed-task_no-debrief_LS')
exp_anx_NY    = he.experiment('mood-and-anxiety_fixed-task_debrief_NY')
exp_ocd_NY    = he.experiment('OCD_fixed-task_debrief_NY')
exp_anx_LS    = he.experiment('mood-and-anxiety_fixed-task_debrief_LS')
exp_ocd_LS    = he.experiment('OCD_fixed-task_debrief_LS')

experiments = [exp_hvs_NY, exp_hvs_LS,exp_hvs_LSnod,exp_anx_NY,exp_ocd_NY,exp_anx_LS,exp_ocd_LS]

sessions_hvs_1 = exp_hvs_NY.get_sessions('all', 1) + exp_hvs_LS.get_sessions('all', 1) + exp_hvs_LSnod.get_sessions('all', 1)
sessions_hvs_3 = exp_hvs_NY.get_sessions('all', 3) + exp_hvs_LS.get_sessions('all', 3) + exp_hvs_LSnod.get_sessions('all', 3) 
sessions_hvs_4 = exp_hvs_NY.get_sessions('all', 4) + exp_hvs_LS.get_sessions('all', 4)  

sessions_hvs_deb_3 = exp_hvs_NY.get_sessions('all', 3) + exp_hvs_LS.get_sessions('all', 3) 

sessions_anxiety_1 = exp_anx_NY.get_sessions('all', 1) + exp_anx_LS.get_sessions('all', 1)
sessions_anxiety_3 = exp_anx_NY.get_sessions('all', 3) + exp_anx_LS.get_sessions('all', 3)
sessions_anxiety_4 = exp_anx_NY.get_sessions('all', 4) + exp_anx_LS.get_sessions('all', 4)

sessions_OCD_1 = exp_ocd_NY.get_sessions('all', 1) + exp_ocd_LS.get_sessions('all', 1) 
sessions_OCD_3 = exp_ocd_NY.get_sessions('all', 3) + exp_ocd_LS.get_sessions('all', 3) 
sessions_OCD_4 = exp_ocd_NY.get_sessions('all', 4) + exp_ocd_LS.get_sessions('all', 4) 

# Split subjects by use of MB at session 3.

sessions_3 = sessions_OCD_3+sessions_hvs_deb_3+sessions_anxiety_3

sessions_to_fit = [session for session in sessions_3 if not hasattr(session, 'fits')]

if sessions_to_fit: # Fit any sessions without fits already saved.
    pp.enable_multiprocessing()
    mc.store_fits_on_sessions(sessions_to_fit, [MF_agent, RL_agent], repeats=50)
    for experiment in experiments:
        experiment.save()

MB_subjects, MF_subjects = mc.two_agent_per_subject_comp(sessions_3, agent_name_A='MF_MBi_bs_ck', return_subs=True)

sessions_hvs_MF_3  = [s for s in sessions_hvs_deb_3  if s.subject_ID in MF_subjects]
sessions_hvs_MF_4  = [s for s in sessions_hvs_4  if s.subject_ID in MF_subjects]

sessions_OCD_MF_3  = [s for s in sessions_OCD_3  if s.subject_ID in MF_subjects]
sessions_OCD_MF_4  = [s for s in sessions_OCD_4  if s.subject_ID in MF_subjects]

sessions_anxiety_MF_3  = [s for s in sessions_anxiety_3  if s.subject_ID in MF_subjects]
sessions_anxiety_MF_4  = [s for s in sessions_anxiety_4  if s.subject_ID in MF_subjects]

# Split subjects by treated / untreated.

sessions_untOCD_1 = exp_ocd_NY.get_sessions('all', 1)
sessions_untOCD_3 = exp_ocd_NY.get_sessions('all', 3)
sessions_untOCD_4 = exp_ocd_NY.get_sessions('all', 4)

sessions_untOCD_MF_3  = [s for s in sessions_untOCD_3  if s.subject_ID in MF_subjects]
sessions_untOCD_MF_4  = [s for s in sessions_untOCD_4  if s.subject_ID in MF_subjects]

sessions_treOCD_1 = exp_ocd_LS.get_sessions('all', 1) 
sessions_treOCD_3 = exp_ocd_LS.get_sessions('all', 3) 
sessions_treOCD_4 = exp_ocd_LS.get_sessions('all', 4)

sessions_treOCD_MF_3  = [s for s in sessions_treOCD_3  if s.subject_ID in MF_subjects]
sessions_treOCD_MF_4  = [s for s in sessions_treOCD_4  if s.subject_ID in MF_subjects]

sessions_untANX_1 = exp_anx_NY.get_sessions('all', 1)
sessions_untANX_3 = exp_anx_NY.get_sessions('all', 3)
sessions_untANX_4 = exp_anx_NY.get_sessions('all', 4)

sessions_untANX_MF_3  = [s for s in sessions_untANX_3  if s.subject_ID in MF_subjects]
sessions_untANX_MF_4  = [s for s in sessions_untANX_4  if s.subject_ID in MF_subjects]

sessions_treANX_1 = exp_anx_LS.get_sessions('all', 1)
sessions_treANX_3 = exp_anx_LS.get_sessions('all', 3)
sessions_treANX_4 = exp_anx_LS.get_sessions('all', 4)

sessions_treANX_MF_3  = [s for s in sessions_treANX_3  if s.subject_ID in MF_subjects]
sessions_treANX_MF_4  = [s for s in sessions_treANX_4  if s.subject_ID in MF_subjects]

#------------------------------------------------------------------------------
# Main figures
#------------------------------------------------------------------------------

def fig_3AD_learning_analysis_OCD():
    '''Generate panels for figure 3A-D showing learning effects in OCD patients.'''
    ylim=[0.5,0.9]
    pl.stay_probability_analysis(sessions_OCD_1, fig_no=2, title='OCD 1', ylim=ylim)
    pl.stay_probability_analysis(sessions_OCD_3, fig_no=3, title='OCD 3', ylim=ylim)

    gc.model_fit_comparison(sessions_OCD_1, sessions_OCD_3, LR_agent, fig_no=4, title='OCD 1 vs 3')

    gc.model_fit_comparison(sessions_OCD_1, sessions_OCD_3, RL_agent, fig_no=5, title='OCD 1 vs 3')

    gc.second_step_RT_comparison(sessions_OCD_1, sessions_OCD_3, fig_no=6)


def fig_3_EH_learning_analysis_mood_anxiety():
    '''Generate panels for figure 3E-H showing learning effects in mood & anxiety patients.'''
    ylim=[0.5,0.9]
    pl.stay_probability_analysis(sessions_anxiety_1, fig_no=2, title='mood & anxiety 1', ylim=ylim)
    pl.stay_probability_analysis(sessions_anxiety_3, fig_no=3, title='mood & anxiety 3', ylim=ylim)

    gc.model_fit_comparison(sessions_anxiety_1, sessions_anxiety_3, LR_agent, fig_no=4, title='mood & anxiety 1 vs 3')

    gc.model_fit_comparison(sessions_anxiety_1, sessions_anxiety_3, RL_agent, fig_no=5, title='mood & anxiety 1 vs 3')

    gc.second_step_RT_comparison(sessions_anxiety_1, sessions_anxiety_3, fig_no=14)


def fig_5AE_debrifing_analysis_OCD():
    '''Generate panels for figure 5A-E showing learning effects in OCD patients.'''
    # Likelihood ratio tests.
    ymax = 40
    mc.two_agent_per_subject_comp(sessions_OCD_MF_3, fig_no=1, title='MF subjects OCD 3', ymax=ymax)
    mc.two_agent_per_subject_comp(sessions_OCD_MF_4, fig_no=2, title='MF subjects OCD 4', ymax=ymax)
    # Stay probabilities.
    ylim = [0.4,1]
    pl.stay_probability_analysis(sessions_OCD_MF_3, fig_no=4, title='OCD 3', ylim=ylim)
    pl.stay_probability_analysis(sessions_OCD_MF_4, fig_no=5, title='OCD 4', ylim=ylim)
    # Logistic regression
    gc.model_fit_comparison(sessions_OCD_MF_3, sessions_OCD_MF_4, agent=LR_agent, fig_no=6, title='OCD 3 vs 4')
    # RL fits.
    gc.model_fit_comparison(sessions_OCD_MF_3, sessions_OCD_MF_4, agent=RL_agent, fig_no=7, title='OCD 3 vs 4')
    # RT analysis
    gc.second_step_RT_comparison(sessions_OCD_MF_3, sessions_OCD_MF_4, fig_no=8)


def fig_5FJ_debrifing_analysis_anxiety():
    '''Generate panels for figure 5F-J showing learning effects in mood & anxiety patients.'''
    # Likelihood ratio tests.
    ymax = 40
    mc.two_agent_per_subject_comp(sessions_anxiety_MF_3, fig_no=1, title='MF subjects mood & anxiety 3', ymax=ymax)
    mc.two_agent_per_subject_comp(sessions_anxiety_MF_4, fig_no=2, title='MF subjects mood & anxiety 4', ymax=ymax)
    # Stay probabilities.
    ylim = [0.4,1]
    pl.stay_probability_analysis(sessions_anxiety_MF_3, fig_no=4, title='mood & anxiety 3', ylim=ylim)
    pl.stay_probability_analysis(sessions_anxiety_MF_4, fig_no=5, title='mood & anxiety 4', ylim=ylim)
    # Logistic regression
    gc.model_fit_comparison(sessions_anxiety_MF_3, sessions_anxiety_MF_4, agent=LR_agent, fig_no=6, title='mood & anxiety 3 vs 4')
    # RL fits.
    gc.model_fit_comparison(sessions_anxiety_MF_3, sessions_anxiety_MF_4, agent=RL_agent, fig_no=7, title='mood & anxiety 3 vs 4')
    # RT analysis
    gc.second_step_RT_comparison(sessions_anxiety_MF_3, sessions_anxiety_MF_4, fig_no=8)

#------------------------------------------------------------------------------
# Supplementary figures
#------------------------------------------------------------------------------

def fig_S7_debriefing_effect_correlations_clinical():
    '''Analyse correlations among different effects of debriefing for healthy volunteers.'''
    gc.debriefing_effect_correlations(sessions_OCD_MF_3, sessions_OCD_MF_4, fig_no=1)
    gc.debriefing_effect_correlations(sessions_anxiety_MF_3, sessions_anxiety_MF_4, fig_no=2)
    
def fig_S9_debriefing_effect_OCD_treatement_comparison():
    '''Plot debriefing effects seperately for the treated (Lisbon) OCD group and untreated (New York) OCD group.'''
    gc.model_fit_comparison(sessions_untOCD_MF_3, sessions_untOCD_MF_4, agent=RL_agent, fig_no=1, title='Untreated OCD 3 vs 4')
    gc.model_fit_comparison(sessions_treOCD_MF_3, sessions_treOCD_MF_4, agent=RL_agent, fig_no=1, title='Untreated OCD 3 vs 4')
    
#------------------------------------------------------------------------------
# Permutation tests
#------------------------------------------------------------------------------

# OCD learning effects

def OCD_lrn():
    '''Test for significant learning effects in OCD patients (i.e. differences between session 1 and 3).'''
    gc.model_fit_test(sessions_OCD_1, sessions_OCD_3, LR_agent, perm_type='within_subject', file_name='LR OCD 1 vs 3', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_OCD_1, sessions_OCD_3, RL_agent, perm_type='within_subject', file_name='RL OCD 1 vs 3', n_true_fit=11, n_perms=5000)

def OCD_lrni():
    '''Test for significant differences in learning effects between OCD and healthy volunteer subjects.'''
    gc.model_fit_interaction_test(sessions_hvs_1, sessions_hvs_3, sessions_OCD_1, sessions_OCD_3, LR_agent, file_name='LR OCD - controls 1,3 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_hvs_1, sessions_hvs_3, sessions_OCD_1, sessions_OCD_3, RL_agent, file_name='RL OCD - controls 1,3 interaction', n_true_fit=11, n_perms=5000)

# Mood and Anxiety learning effects

def ANX_lrn():
    '''Test for significant learning effects in mood and anxiety patients (i.e. differences between session 1 and 3).'''
    gc.model_fit_test(sessions_anxiety_1, sessions_anxiety_3, LR_agent, perm_type='within_subject', file_name='LR anxiety 1 vs 3', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_anxiety_1, sessions_anxiety_3, RL_agent, perm_type='within_subject', file_name='RL anxiety 1 vs 3', n_true_fit=11, n_perms=5000)

def ANX_lrni():
    '''Test for significant differences in learning effects between M&A and healthy volunteer subjects.'''
    gc.model_fit_interaction_test(sessions_hvs_1, sessions_hvs_3, sessions_anxiety_1, sessions_anxiety_3, LR_agent, file_name='LR anxiety - controls 1,3 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_hvs_1, sessions_hvs_3, sessions_anxiety_1, sessions_anxiety_3, RL_agent, file_name='RL anxiety - controls 1,3 interaction', n_true_fit=11, n_perms=5000)

# OCD debriefing effects

def OCD_dbr():
    '''Test for significant debriefing effects in OCD patients (i.e. differences between session 3 and 4 in subjects who are MF at session 3).'''
    gc.model_fit_test(sessions_OCD_MF_3, sessions_OCD_MF_4, LR_agent, perm_type='within_subject', file_name='LR OCD MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_OCD_MF_3, sessions_OCD_MF_4, RL_agent, perm_type='within_subject', file_name='RL OCD MF 3 vs 4', n_true_fit=11, n_perms=5000)

def OCD_dbri():
    '''Test for significant differences in debriefing effects between OCD and healthy volunteer subjects.'''
    gc.model_fit_interaction_test(sessions_hvs_MF_3, sessions_hvs_MF_4, sessions_OCD_MF_3, sessions_OCD_MF_4, LR_agent, file_name='LR OCD - controls MF 3,4 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_hvs_MF_3, sessions_hvs_MF_4, sessions_OCD_MF_3, sessions_OCD_MF_4, RL_agent, file_name='RL OCD - controls MF 3,4 interaction', n_true_fit=11, n_perms=5000)

# Mood and Anxiety debriefing effects

def ANX_dbr():
    '''Test for significant debriefing effects in N&A patients (i.e. differences between session 3 and 4 in subjects who are MF at session 3).'''
    gc.model_fit_test(sessions_anxiety_MF_3, sessions_anxiety_MF_4, LR_agent, perm_type='within_subject', file_name='LR anxiety MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_anxiety_MF_3, sessions_anxiety_MF_4, RL_agent, perm_type='within_subject', file_name='RL anxiety MF 3 vs 4', n_true_fit=11, n_perms=5000)

def ANX_dbri():
    '''Test for significant differences in debriefing effects between M&A and healthy volunteer subjects.'''
    gc.model_fit_interaction_test(sessions_hvs_MF_3, sessions_hvs_MF_4, sessions_anxiety_MF_3, sessions_anxiety_MF_4, LR_agent, file_name='LR anxiety - controls MF 3,4 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_hvs_MF_3, sessions_hvs_MF_4, sessions_anxiety_MF_3, sessions_anxiety_MF_4, RL_agent, file_name='RL anxiety - controls MF 3,4 interaction', n_true_fit=11, n_perms=5000)

# treated vs untreated tests

def trt_OCD_lrn():
    '''Test for significant differences in learning effects between OCD subjects on and off medication.'''
    gc.model_fit_interaction_test(sessions_untOCD_1, sessions_untOCD_3, sessions_treOCD_1, sessions_treOCD_3, LR_agent, file_name='LR untOCD vs treOCD 1,3 interaction', n_true_fit=11, n_perms=5000)  
    gc.model_fit_interaction_test(sessions_untOCD_1, sessions_untOCD_3, sessions_treOCD_1, sessions_treOCD_3, RL_agent, file_name='RL untOCD vs treOCD 1,3 interaction', n_true_fit=11, n_perms=5000)  

def trt_OCD_dbr():
    '''Test for significant differences in debriefing effects between OCD subjects on and off medication.'''
    gc.model_fit_interaction_test(sessions_untOCD_MF_3, sessions_untOCD_MF_4, sessions_treOCD_MF_3, sessions_treOCD_MF_4, LR_agent, file_name='LR untOCD vs treOCD 3,4 interaction', n_true_fit=11, n_perms=5000)  
    gc.model_fit_interaction_test(sessions_untOCD_MF_3, sessions_untOCD_MF_4, sessions_treOCD_MF_3, sessions_treOCD_MF_4, RL_agent, file_name='RL untOCD vs treOCD 3,4 interaction', n_true_fit=11, n_perms=5000)  

def trt_ANX_lrn():
    '''Test for significant differences in learning effects between ANX subjects on and off medication.'''
    gc.model_fit_interaction_test(sessions_untANX_1, sessions_untANX_3, sessions_treANX_1, sessions_treANX_3, LR_agent, file_name='LR untANX vs treANX 1,3 interaction', n_true_fit=11, n_perms=5000)  
    gc.model_fit_interaction_test(sessions_untANX_1, sessions_untANX_3, sessions_treANX_1, sessions_treANX_3, RL_agent, file_name='RL untANX vs treANX 1,3 interaction', n_true_fit=11, n_perms=5000)  

def trt_ANX_dbr():
    '''Test for significant differences in debriefing effects between ANX subjects on and off medication.'''
    gc.model_fit_interaction_test(sessions_untANX_MF_3, sessions_untANX_MF_4, sessions_treANX_MF_3, sessions_treANX_MF_4, LR_agent, file_name='LR untANX vs treANX 3,4 interaction', n_true_fit=11, n_perms=5000)  
    gc.model_fit_interaction_test(sessions_untANX_MF_3, sessions_untANX_MF_4, sessions_treANX_MF_3, sessions_treANX_MF_4, RL_agent, file_name='RL untANX vs treANX 3,4 interaction', n_true_fit=11, n_perms=5000)  
    