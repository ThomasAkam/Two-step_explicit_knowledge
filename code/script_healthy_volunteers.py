'''Script to generate figures and run permutation tests for healthy volunteers.'''

from Two_step import  mc, lr, gc, he, rl, pl, pp, sm

pp.enable_multiprocessing() # Use multiprocessing for model fitting.

#------------------------------------------------------------------------------
# Logistic regression and RL models.
#------------------------------------------------------------------------------

LR_agent = lr.config_log_reg()
MF_agent = rl.MF()
MF_MB_agent = rl.MF_MB()    # RL agent used for changing task.
MF_MBi_agent = rl.MF_MBi()  # RL agent used for fixed task.

#------------------------------------------------------------------------------
# Import data
#------------------------------------------------------------------------------

# The first time the script is imported the fits needed to identify subjects  
# who are MF and MB at session 3 are calculated which will take a while. These
# fits are then saved making subsequent imports much faster.

# Fixed task  -----------------------------------------------------------------

exp_fixed_deb_LS    = he.experiment('healthy-volunteers_fixed-task_debrief_LS')
exp_fixed_nod_LS    = he.experiment('healthy-volunteers_fixed-task_no-debrief_LS')
exp_fixed_deb_NY = he.experiment('healthy-volunteers_fixed-task_debrief_NY')

sessions_fixed_1   = exp_fixed_deb_LS.get_sessions('all', 1)       + exp_fixed_nod_LS.get_sessions('all', 1)        + exp_fixed_deb_NY.get_sessions('all', 1) 
sessions_fixed_3   = exp_fixed_deb_LS.get_sessions('all', 3)       + exp_fixed_nod_LS.get_sessions('all', 3)        + exp_fixed_deb_NY.get_sessions('all', 3)      
sessions_fixed_4   = exp_fixed_deb_LS.get_sessions('all', 4)       + exp_fixed_nod_LS.get_sessions('all', 4)
sessions_fixed_123 = exp_fixed_deb_LS.get_sessions('all', [1,2,3]) + exp_fixed_nod_LS.get_sessions('all', [1,2,3])  + exp_fixed_deb_NY.get_sessions('all', [1,2,3]) 

sessions_fixed_deb_3 = exp_fixed_deb_LS.get_sessions('all', 3) + exp_fixed_deb_NY.get_sessions('all', 3)
sessions_fixed_deb_4 = exp_fixed_deb_LS.get_sessions('all', 4) + exp_fixed_deb_NY.get_sessions('all', 4)
sessions_fixed_nod_3 = exp_fixed_nod_LS.get_sessions('all', 3)
sessions_fixed_nod_4 = exp_fixed_nod_LS.get_sessions('all', 4)

# Select subjects who are not model-based in session 3.
    
if not hasattr(sessions_fixed_3[0], 'fits'): # Store fits if not already stored.
    mc.store_fits_on_sessions(sessions_fixed_3+sessions_fixed_4, [MF_agent, MF_MBi_agent], repeats=50)
    for experiment in (exp_fixed_deb_LS, exp_fixed_nod_LS, exp_fixed_deb_NY):
        experiment.save() # Save sessions to .pkl file.

MB_subjects, MF_subjects = mc.two_agent_per_subject_comp(sessions_fixed_3, agent_name_A='MF_MBi_bs_ck', return_subs=True)
sessions_fixed_nod_MF_3  = [s for s in sessions_fixed_nod_3  if s.subject_ID in MF_subjects]
sessions_fixed_nod_MF_4  = [s for s in sessions_fixed_nod_4  if s.subject_ID in MF_subjects]
sessions_fixed_deb_MF_3  = [s for s in sessions_fixed_deb_3  if s.subject_ID in MF_subjects]
sessions_fixed_deb_MF_4  = [s for s in sessions_fixed_deb_4  if s.subject_ID in MF_subjects]

# Split sessions by Lisbon and New York subjects.

sessions_1_LS   = exp_fixed_deb_LS.get_sessions('all', 1) + exp_fixed_nod_LS.get_sessions('all', 1)
sessions_3_LS   = exp_fixed_deb_LS.get_sessions('all', 3) + exp_fixed_nod_LS.get_sessions('all', 3)
sessions_3_deb_LS = exp_fixed_deb_LS.get_sessions('all', 3)

sessions_4_LS   = exp_fixed_deb_LS.get_sessions('all', 4)

sessions_1_NY = exp_fixed_deb_NY.get_sessions('all', 1)
sessions_3_NY = exp_fixed_deb_NY.get_sessions('all', 3)
sessions_4_NY = exp_fixed_deb_NY.get_sessions('all', 4)

sessions_MF_3_LS  = [s for s in sessions_3_deb_LS if s.subject_ID in MF_subjects]
sessions_MF_4_LS  = [s for s in sessions_4_LS     if s.subject_ID in MF_subjects]

sessions_MF_3_NY  = [s for s in sessions_3_NY if s.subject_ID in MF_subjects]
sessions_MF_4_NY  = [s for s in sessions_4_NY if s.subject_ID in MF_subjects]

# Changing task ---------------------------------------------------------------

exp_change_deb   = he.experiment('healthy-volunteers_changing-task_debrief_LS')
exp_change_nod   = he.experiment('healthy-volunteers_changing_no-debrief_LS')

sessions_change_1   = exp_change_deb.get_sessions('all', 1)       + exp_change_nod.get_sessions('all', 1)
sessions_change_3   = exp_change_deb.get_sessions('all', 3)       + exp_change_nod.get_sessions('all', 3)
sessions_change_4   = exp_change_deb.get_sessions('all', 4)       + exp_change_nod.get_sessions('all', 4)
sessions_change_123 = exp_change_deb.get_sessions('all', [1,2,3]) + exp_change_nod.get_sessions('all', [1,2,3])

sessions_change_deb_3 = exp_change_deb.get_sessions('all', 3)
sessions_change_nod_3 = exp_change_nod.get_sessions('all', 3)
sessions_change_deb_4 = exp_change_deb.get_sessions('all', 4)
sessions_change_nod_4 = exp_change_nod.get_sessions('all', 4)

# Select subjects who are not model-based in session 3.

if not hasattr(sessions_change_3[0], 'fits'): # Store fits if not already stored.
    mc.store_fits_on_sessions(sessions_change_3+sessions_change_4, [MF_agent, MF_MB_agent], repeats=50)
    exp_change_deb.save() # Save sessions to .pkl file.
    exp_change_nod.save()
        
MB_subjects, MF_subjects = mc.two_agent_per_subject_comp(sessions_change_3, agent_name_A='MF_MB_bs_ck', return_subs=True)
sessions_change_nod_MF_3  = [s for s in sessions_change_nod_3  if s.subject_ID in MF_subjects]
sessions_change_nod_MF_4  = [s for s in sessions_change_nod_4  if s.subject_ID in MF_subjects]
sessions_change_deb_MF_3  = [s for s in sessions_change_deb_3  if s.subject_ID in MF_subjects]
sessions_change_deb_MF_4  = [s for s in sessions_change_deb_4  if s.subject_ID in MF_subjects]

# Slow paced task -------------------------------------------------------------

exp_slow = he.experiment('healthy-volunteers_slow-paced-task_debrief_LS')

sessions_slow_1 = exp_slow.get_sessions('all', 1)
sessions_slow_3 = exp_slow.get_sessions('all', 3)
sessions_slow_4 = exp_slow.get_sessions('all', 4)

# Select subjects who are not model-based in session 3.

if not hasattr(sessions_slow_3[0], 'fits'): # Store fits if not already stored.
    mc.store_fits_on_sessions(sessions_slow_3+sessions_slow_4, [MF_agent, MF_MBi_agent], repeats=50)
    exp_slow.save() # Save sessions to .pkl file.

MB_subjects, MF_subjects = mc.two_agent_per_subject_comp(sessions_slow_3, agent_name_A='MF_MBi_bs_ck', return_subs=True)

sessions_slow_MF_3  = [s for s in sessions_slow_3  if s.subject_ID in MF_subjects]
sessions_slow_MF_4  = [s for s in sessions_slow_4  if s.subject_ID in MF_subjects]

#------------------------------------------------------------------------------
# Main figures 
#------------------------------------------------------------------------------

def fig_2_learning_effects_fixed_task():
    '''Generate panels for figure 2 showing learning effects on fixed task.'''
    # Stay probability analysis.
    pl.stay_probability_analysis(sessions_fixed_1, fig_no=1, title='Fixed 1')
    pl.stay_probability_analysis(sessions_fixed_3, fig_no=2, title='Fixed 3')
    pl.outcome_effect_timecourse(sessions_fixed_1, fig_no=3)
    # Stay probability logistic regression.
    gc.model_fit_comparison(sessions_fixed_1, sessions_fixed_3, LR_agent, fig_no=4, title='Fixed 1 vs 3') 
    # RT analysis.
    gc.second_step_RT_comparison(sessions_fixed_1, sessions_fixed_3, fig_no=5)
    # RL model fits.
    gc.model_fit_comparison(sessions_fixed_1, sessions_fixed_3, MF_MBi_agent, fig_no=9, title='Fixed 1 vs 3') 


def fig_4_debriefing_analysis_fixed_task():
    '''Generate panels for figure 4 showing debriefing effects on fixed task.'''
    # Likelihood ratio tests.
    mc.two_agent_per_subject_comp(sessions_fixed_nod_MF_3, fig_no=1, title='MF subjects no-debriefing 3', ymax=40)
    mc.two_agent_per_subject_comp(sessions_fixed_nod_MF_4, fig_no=2, title='MF subjects no-debriefing 4', ymax=40)
    mc.two_agent_per_subject_comp(sessions_fixed_deb_MF_3, fig_no=3, title='MF subjects debriefing 3'   , ymax=40)
    mc.two_agent_per_subject_comp(sessions_fixed_deb_MF_4, fig_no=4, title='MF subjects debriefing 4'   , ymax=40)
    # Stay probabilities.
    pl.stay_probability_analysis(sessions_fixed_nod_MF_3, fig_no=7, title='No-debriefing 3')
    pl.stay_probability_analysis(sessions_fixed_nod_MF_4, fig_no=8, title='No-debriefing 4')
    pl.stay_probability_analysis(sessions_fixed_deb_MF_3, fig_no=9, title='Debriefing 3')
    pl.stay_probability_analysis(sessions_fixed_deb_MF_4, fig_no=10, title='Debriefing 4')
    # Logistic regression
    gc.model_fit_comparison(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, agent=LR_agent, 
                            fig_no=11, title='fixed debriefing 3 vs 4')
    gc.model_fit_comparison(sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, agent=LR_agent, 
                            fig_no=12, title='fixed no-debriefing 3 vs 4')
    # RT analysis.
    gc.second_step_RT_comparison(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, fig_no=13)
    gc.second_step_RT_comparison(sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, fig_no=14)
    # RL fits.
    gc.model_fit_comparison(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, agent=MF_MBi_agent, 
                            fig_no=15, title='fixed debriefing 3 vs 4')
    gc.model_fit_comparison(sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, agent=MF_MBi_agent, 
                            fig_no=16, title='fixed no-debriefing 3 vs 4')

#------------------------------------------------------------------------------
# Supplementary figures
#------------------------------------------------------------------------------

def fig_S1_invalid_press_analysis():
    '''Analysis of the rate of presses to invalid options.'''
    pl.invalid_press_analysis(sessions_fixed_1, fig_no=1)
    pl.invalid_press_analysis(sessions_fixed_3, fig_no=2)
    pl.invalid_press_analysis(sessions_fixed_deb_4, fig_no=3)
    pl.invalid_press_analysis(sessions_fixed_nod_4, fig_no=4)


def fig_S2_fixed_task_model_comparison():
    '''BIC score based model comparison for fixed task data.'''
    # Comparison of model-free, model-based and mixture agents.
    mc.BIC_model_comparison(sessions_fixed_123, agents=[rl.MF(['bs','ck']),rl.MBi(['bs','ck']),rl.MF_MBi(['bs','ck'])], fig_no=1)
    # Comparison of mixture agents with and without bias and perseveration.
    mc.BIC_model_comparison(sessions_fixed_123, agents=[rl.MF_MBi(['bs','ck']), rl.MF_MBi(['bs']), rl.MF_MBi(['ck'])] , fig_no=2)
    # Comparison with incorrect model agents.
    agents=[rl.MF(['bs','ck']),rl.MBi(['bs','ck']),rl.MF_MBi(['bs','ck']),
            rl.IM_DV(['bs','ck']), rl.IM_TDLR(['bs','ck']), rl.IM_US(['bs','ck'])]
    mc.BIC_model_comparison(sessions_fixed_1, agents, fig_no=3)
    mc.BIC_model_comparison(sessions_fixed_3, agents, fig_no=4)
    
    
def fig_S3_choice_vs_implicit_learning_correlations():
    '''Plot correlations amound choice and implicit measures of task structure learning'''
    gc.MB_motor_prediction_correlations(sessions_fixed_1, MF_MBi_agent, LR_agent)
    gc.MB_motor_prediction_correlations(sessions_fixed_3, MF_MBi_agent, LR_agent)
    gc.MB_motor_prediction_correlations(sessions_fixed_deb_4, MF_MBi_agent, LR_agent)
 
    
def fig_S4_learning_analysis_slow():
    '''Analysis of learning effects in slow pace control experiment.'''
    ylim=[0.5,0.9]
    pl.stay_probability_analysis(sessions_slow_1, fig_no=2, title='slow 1', ylim=ylim)
    pl.stay_probability_analysis(sessions_slow_3, fig_no=3, title='slow 3', ylim=ylim)
    pl.outcome_effect_timecourse(sessions_slow_1, fig_no=4)
    gc.model_fit_comparison(sessions_slow_1, sessions_slow_3, LR_agent, fig_no=4, title='slow 1 vs 3')
    gc.model_fit_comparison(sessions_slow_1, sessions_slow_3, MF_MBi_agent, fig_no=5, title='slow 1 vs 3')
    gc.second_step_RT_comparison(sessions_slow_1, sessions_slow_3, fig_no=6)


def fig_S5_debrifing_analysis_slow():
    '''Analysis of debreifing effects in slow pace control experiment.'''
    # Likelihood ratio tests.
    mc.two_agent_per_subject_comp(sessions_slow_MF_3, fig_no=1, title='MF subjects slow 3', ymax=20)
    mc.two_agent_per_subject_comp(sessions_slow_MF_4, fig_no=2, title='MF subjects slow 4', ymax=20)
    # Stay probabilities.
    pl.stay_probability_analysis(sessions_slow_MF_3, fig_no=3, title='slow 3', ylim=[0.4,1])
    pl.stay_probability_analysis(sessions_slow_MF_4, fig_no=4, title='slow 4', ylim=[0.4,1])
    # Logistic regression
    gc.model_fit_comparison(sessions_slow_MF_3, sessions_slow_MF_4, agent=LR_agent, fig_no=5, title='slow 3 vs 4')
    # RL fits.
    gc.model_fit_comparison(sessions_slow_MF_3, sessions_slow_MF_4, agent=MF_MBi_agent, fig_no=6, title='slow 3 vs 4')
    # RT analysis
    gc.second_step_RT_comparison(sessions_slow_MF_3, sessions_slow_MF_4, fig_no=7)


def fig_S6_simulated_data_stay_probabilities():
    '''Plot stay probabilities for real and simulated data'''
    sessions_sim_1 = sm.simulate_sessions_from_fit(sessions_fixed_1, MF_MBi_agent)
    sessions_sim_3 = sm.simulate_sessions_from_fit(sessions_fixed_3, MF_MBi_agent)
    sessions_sim_4 = sm.simulate_sessions_from_fit(sessions_fixed_deb_4, MF_MBi_agent)
    pl.stay_probability_analysis(sessions_fixed_1    , fig_no=1, title='Session 1 data')
    pl.stay_probability_analysis(sessions_fixed_3    , fig_no=2, title='Session 1 data')
    pl.stay_probability_analysis(sessions_fixed_deb_4, fig_no=3, title='Session 4 data')
    pl.stay_probability_analysis(sessions_sim_1, fig_no=4, title='Session 1 simulation')
    pl.stay_probability_analysis(sessions_sim_3, fig_no=5, title='Session 1 simulation')
    pl.stay_probability_analysis(sessions_sim_4, fig_no=6, title='Session 4 simulation')
    
    
def fig_S7_debriefing_effect_correlations_HVs():
    '''Analyse correlations among different effects of debriefing for healthy volunteers.'''
    gc.debriefing_effect_correlations(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, fig_no=1)
    
    
def fig_S8_RL_model_paramter_recovery():
    '''Test the accuracy with which RL model parameters can be recovered from simulated data'''
    sm.parameter_recovery(sessions_fixed_1, MF_MBi_agent, fig_no=1)
    sm.parameter_recovery(sessions_fixed_3, MF_MBi_agent, fig_no=2)
    sm.parameter_recovery(sessions_fixed_deb_4, MF_MBi_agent, fig_no=3)
    
    
def fig_S10_learning_effects_changing_task():
    '''Analysis of learning effects in the task with changing transition probabilities.'''
    # Stay probability analysis.
    pl.stay_probability_analysis(sessions_change_1, fig_no=1, title='Change 1')
    pl.stay_probability_analysis(sessions_change_3, fig_no=2, title='Change 3')
    # Stay probability logistic regression.
    gc.model_fit_comparison(sessions_change_1, sessions_change_3, LR_agent, fig_no=3, title='Fixed 1 vs 3') 
    # RT analysis.
    gc.second_step_RT_comparison(sessions_change_1, sessions_change_3, fig_no=4)
    # RL model fits.
    gc.model_fit_comparison(sessions_change_1, sessions_change_3, MF_MBi_agent, fig_no=5, title='Fixed 1 vs 3') 
    # RL model comparison.
    mc.BIC_model_comparison(sessions_change_123, agents=[rl.MF(['bs','ck']),rl.MBi(['bs','ck']),rl.MF_MBi(['bs','ck'])], fig_no=6)
    mc.BIC_model_comparison(sessions_change_123, agents=[rl.MF_MBi(['bs','ck']), rl.MF_MBi(['bs']), rl.MF_MBi(['ck'])] , fig_no=7)
    

def fig_S11_debriefing_analysis_changing_task():
    '''Analysis of debriefing effects in the task with changing transition probabilities.'''
    # Likelihood ratio tests.
    mc.two_agent_per_subject_comp(sessions_change_nod_MF_3, fig_no=1, title='MF subjects no-debriefing 3', ymax=15, agent_name_A='MF_MB_bs_ck')
    mc.two_agent_per_subject_comp(sessions_change_nod_MF_4, fig_no=2, title='MF subjects no-debriefing 4', ymax=15, agent_name_A='MF_MB_bs_ck')
    mc.two_agent_per_subject_comp(sessions_change_deb_MF_3, fig_no=3, title='MF subjects debriefing 3'   , ymax=15, agent_name_A='MF_MB_bs_ck')
    mc.two_agent_per_subject_comp(sessions_change_deb_MF_4, fig_no=4, title='MF subjects debriefing 4'   , ymax=15, agent_name_A='MF_MB_bs_ck')
    # Stay probabilities.
    pl.stay_probability_analysis(sessions_change_nod_MF_3, fig_no=7, title='No-debriefing 3')
    pl.stay_probability_analysis(sessions_change_nod_MF_4, fig_no=8, title='No-debriefing 4')
    pl.stay_probability_analysis(sessions_change_deb_MF_3, fig_no=9, title='Debriefing 3')
    pl.stay_probability_analysis(sessions_change_deb_MF_4, fig_no=10, title='Debriefing 4')
    # Logistic regression
    gc.model_fit_comparison(sessions_change_deb_MF_3, sessions_change_deb_MF_4, agent=LR_agent, 
                            fig_no=11, title='full debriefing 3 vs 4')
    gc.model_fit_comparison(sessions_change_nod_MF_3, sessions_change_nod_MF_4, agent=LR_agent, 
                            fig_no=12, title='full no-debriefing 3 vs 4')
    # RL fits.
    gc.model_fit_comparison(sessions_change_deb_MF_3, sessions_change_deb_MF_4, agent=MF_MB_agent, 
                            fig_no=13, title='full debriefing 3 vs 4')
    gc.model_fit_comparison(sessions_change_nod_MF_3, sessions_change_nod_MF_4, agent=MF_MB_agent, 
                            fig_no=14, title='full no-debriefing 3 vs 4')
    # RT analysis.
    gc.second_step_RT_comparison(sessions_change_deb_MF_3, sessions_change_deb_MF_4, fig_no=15)
    gc.second_step_RT_comparison(sessions_change_nod_MF_3, sessions_change_nod_MF_4, fig_no=16)
    
#------------------------------------------------------------------------------
# Permutation tests.
#------------------------------------------------------------------------------
 
# Fixed task
    
def fixed_task_learning_effects():
    '''Test for significant effects of learning in the fixed task (i.e. differences between session 1 and 3).'''
    lr.predictor_significance_test(sessions_fixed_1, LR_agent, n_perms=5000, file_name='LR fixed 1 pred sig')
    gc.model_fit_test(sessions_fixed_1, sessions_fixed_3, LR_agent    , perm_type='within_subject', file_name='LR fixed 1 vs 3', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_fixed_1, sessions_fixed_3, MF_MBi_agent, perm_type='within_subject', file_name='RL fixed 1 vs 3', n_true_fit=11, n_perms=5000)

def fixed_task_debriefing_effects():
    '''Test for significant differences beween session 3 and 4 in the debriefing group on the fixed task.'''
    gc.model_fit_test(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, LR_agent    , perm_type='within_subject', file_name='LR fixed deb MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, MF_MBi_agent, perm_type='within_subject', file_name='RL fixed deb MF 3 vs 4', n_true_fit=11, n_perms=5000)

def fixed_task_no_debriefing_effects():
    '''Test for significant differences beween session 3 and 4 in the no-debriefing group on the fixed task.'''
    gc.model_fit_test(sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, LR_agent    , perm_type='within_subject', file_name='LR fixed nod MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, MF_MBi_agent, perm_type='within_subject', file_name='RL fixed nod MF 3 vs 4', n_true_fit=11, n_perms=5000)

def fixed_task_debriefing_interaction():
    '''Test for significant differences between the debriefing and no-debriefing groups wrt the behavioural change between session 3 and 4.'''
    gc.model_fit_interaction_test(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, LR_agent    , file_name='LR fixed deb MF interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_fixed_deb_MF_3, sessions_fixed_deb_MF_4, sessions_fixed_nod_MF_3, sessions_fixed_nod_MF_4, MF_MBi_agent, file_name='RL fixed deb MF interaction', n_true_fit=11, n_perms=5000)

# Changing task

def changing_task_learning_effects():
    '''Test for significant effects of learning in the changing task (i.e. differences between session 1 and 3).'''
    lr.predictor_significance_test(sessions_change_1, LR_agent, n_perms=5000, file_name='LR change 1 pred sig')
    gc.model_fit_test(sessions_change_1, sessions_change_3, LR_agent  ,  perm_type='within_subject', file_name='LR change 1 vs 3', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_change_1, sessions_change_3, MF_MB_agent, perm_type='within_subject', file_name='RL change 1 vs 3', n_true_fit=11, n_perms=5000)

def changing_task_debriefing_effects():
    '''Test for significant differences beween session 3 and 4 in the debriefing group on the changing task.'''
    gc.model_fit_test(sessions_change_deb_MF_3, sessions_change_deb_MF_4, LR_agent   , perm_type='within_subject', file_name='LR change deb MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_change_deb_MF_3, sessions_change_deb_MF_4, MF_MB_agent, perm_type='within_subject', file_name='RL change deb MF 3 vs 4', n_true_fit=11, n_perms=5000)

def changing_task_no_debriefing_effects():
    '''Test for significant differences beween session 3 and 4 in the no-debriefing group on the changing task.'''
    gc.model_fit_test(sessions_change_nod_MF_3, sessions_change_nod_MF_4, LR_agent   , perm_type='within_subject', file_name='LR change nod MF 3 vs 4', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_change_nod_MF_3, sessions_change_nod_MF_4, MF_MB_agent, perm_type='within_subject', file_name='RL change nod MF 3 vs 4', n_true_fit=11, n_perms=5000)

def changing_task_debriefing_interaction():
    '''Test for significant differences between the debriefing and no-debriefing groups wrt the behavioural change between session 3 and 4.'''
    gc.model_fit_interaction_test(sessions_change_deb_MF_3, sessions_change_deb_MF_4, sessions_change_nod_MF_3, sessions_change_nod_MF_4, LR_agent   , file_name='LR change deb MF interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_change_deb_MF_3, sessions_change_deb_MF_4, sessions_change_nod_MF_3, sessions_change_nod_MF_4, MF_MB_agent, file_name='RL change deb MF interaction', n_true_fit=11, n_perms=5000)
    
# Tests for differences between Lisbon and New York subjects.

def LSNY_learning_effects_interaction():
    ''' Test for significant differences in learning effects between Lisbon and NY subjects.'''
    gc.model_fit_interaction_test(sessions_1_LS, sessions_3_LS, sessions_1_NY, sessions_3_NY, LR_agent    , file_name='LR LS-NY 1,3 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_1_LS, sessions_3_LS, sessions_1_NY, sessions_3_NY, MF_MBi_agent, file_name='RL LS-NY 1,3 interaction', n_true_fit=11, n_perms=5000)

def LSNY_debriefing_effects_interaction():
    ''' Test for significant differences in debriefing effects between Lisbon and NY subjects.'''
    gc.model_fit_interaction_test(sessions_MF_3_LS, sessions_MF_4_LS, sessions_MF_3_NY, sessions_MF_4_NY, LR_agent    , file_name='LR LS-NY MF 3,4 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_MF_3_LS, sessions_MF_4_LS, sessions_MF_3_NY, sessions_MF_4_NY, MF_MBi_agent, file_name='RL LS-NY MF 3,4 interaction', n_true_fit=11, n_perms=5000)

# Tests for differences between fixed and changing task.

def FvC_session_1():
    '''Test for significant difference between fixed and changing task at session 1.'''
    gc.model_fit_test(sessions_fixed_1, sessions_change_1, LR_agent   , perm_type='within_subject', file_name='LR fixed v change 1', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_fixed_1, sessions_change_1, MF_MB_agent, perm_type='within_subject', file_name='RL fixed v change 1', n_true_fit=11, n_perms=5000)

def FvC_session_3():
    '''Test for significant difference between fixed and changing task at session 3.'''
    gc.model_fit_test(sessions_fixed_3, sessions_change_3, LR_agent   , perm_type='within_subject', file_name='LR fixed v change 3', n_true_fit=11, n_perms=5000)
    gc.model_fit_test(sessions_fixed_3, sessions_change_3, MF_MB_agent, perm_type='within_subject', file_name='RL fixed v change 3', n_true_fit=11, n_perms=5000)

def FvC_learning_effects_interaction():
    '''Test for significant effects of learning in the changing task (i.e. differences between session 1 and 3).'''
    gc.model_fit_interaction_test(sessions_fixed_1, sessions_fixed_3, sessions_change_1, sessions_change_3, LR_agent   , file_name='LR Fixed-change 1,3 interaction', n_true_fit=11, n_perms=5000)
    gc.model_fit_interaction_test(sessions_fixed_1, sessions_fixed_3, sessions_change_1, sessions_change_3, MF_MB_agent, file_name='RL Fixed-change 1,3 interaction', n_true_fit=11, n_perms=5000)
