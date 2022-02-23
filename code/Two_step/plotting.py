''' Plotting and analysis functions.'''

import pylab as plt
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import sem
import seaborn as sns
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.proportion import proportions_ztest
from . import utility as ut

#----------------------------------------------------------------------------------
# Various analyses.
#----------------------------------------------------------------------------------

def session_plot(session, show_TO=False, ylabel=True):
    'Plot of choice moving average and reward block structure for single session.'
    choices, transitions, second_steps, outcomes = session.unpack_trial_data(dtype=bool)
    second_steps = second_steps * 1.1-0.05
    mov_ave = ut.exp_mov_ave(choices, initValue=0.5)

    if hasattr(session, 'blocks'):
        for i in range(len(session.blocks['start_trials'])):
            y = [0.9,0.5,0.1][session.blocks['reward_states'][i]]  # y position coresponding to reward state.
            x = [session.blocks['start_trials'][i], session.blocks['end_trials'][i]]
            if session.blocks['transition_states'][i]:
                plt.plot(x, [y,y], 'orange', linewidth=2)
            else:
                y = 1 - y  # Invert y position if transition is inverted.
                plt.plot(x, [y,y], 'purple', linewidth=2)

    plt.plot(mov_ave,'k.-', markersize=3)    

    if show_TO:
        def symplot(y,guard,symbol):
            x_ind = np.where(guard)[0]
            plt.plot(x_ind,y[x_ind],symbol, markersize=5)
        symplot(second_steps,  transitions &  outcomes,'ob' )
        symplot(second_steps,  transitions & ~outcomes,'xb')
        symplot(second_steps, ~transitions &  outcomes,'og')
        symplot(second_steps, ~transitions & ~outcomes,'xg')  
    plt.plot([0,len(choices)],[0.75,0.75],'--k')
    plt.plot([0,len(choices)],[0.25,0.25],'--k')

    plt.xlabel('Trial Number')
    plt.yticks([0,0.5,1])
    plt.ylim(-0.1, 1.1)
    plt.xlim(0,len(choices))
    if ylabel:plt.ylabel('Choice moving average')

def block_length_distribution(sessions, fig_no=1, n_bins=20, non_neutral=True):
    'Evalute the distribution of block lengths.'
    block_lengths = []
    for session in sessions:
        start_trials = session.blocks['start_trials']
        ses_block_lens = np.array(start_trials[1:]) - np.array(start_trials[:-1])
        if non_neutral:
            not_neutral = session.blocks['reward_states'][:-1] != 1
            ses_block_lens = ses_block_lens[not_neutral]
        block_lengths.append(ses_block_lens)
    block_lengths = np.hstack(block_lengths)
    plt.figure(fig_no).clf()
    plt.hist(block_lengths, n_bins)
    plt.xlabel('Block length')
    plt.ylabel('Count')
    print('Mean: {}'.format(np.mean(block_lengths)))
    print('SD  : {}'.format(np.std(block_lengths)))
    print('Min : {}'.format(np.min(block_lengths)))
    print('Max : {}'.format(np.max(block_lengths)))

def block_number_distribution(sessions, fig_no=1):
    '''Evalute the distribtion of number of blocks completed across subjects'''
    subject_IDs = [session.subject_ID for session in sessions]
    n_blocks = [] 
    for sID in subject_IDs:
        subject_sessions = [s for s in sessions if s.subject_ID == sID]
        n_blocks.append(sum([len(s.blocks['end_trials']) for s in subject_sessions]))
    plt.figure(fig_no, figsize=[4,3]).clf()
    plt.hist(n_blocks, 10)
    plt.xlabel('Number of blocks')
    plt.ylabel('Count (subjects)')
    plt.tight_layout()
    print('Mean: {}'.format(np.mean(n_blocks)))
    print('SD  : {}'.format(np.std(n_blocks)))
    print('Min : {}'.format(np.min(n_blocks)))
    print('Max : {}'.format(np.max(n_blocks))) 

#----------------------------------------------------------------------------------
# Event timing analyses.
#----------------------------------------------------------------------------------

def second_step_reaction_times(sessions, fig_no=1, col='b', return_med=False):
    subject_IDs = [session.subject_ID for session in sessions]
    assert len(subject_IDs) == len(set(subject_IDs)), 'Function requires single session per subject.'
    common_RTs = {}
    rare_RTs   = {}
    for session in sessions:
        ssRTs =  _latencies(session.times['second_step'], session.times['ss_action'])
        common_trans = (session.trial_data['transitions'] ==
                        session.blocks['trial_trans_state'])
        common_RTs[session.subject_ID] = ssRTs[ common_trans]
        rare_RTs[  session.subject_ID] = ssRTs[~common_trans]
    all_common_RT = np.hstack([x for x in common_RTs.values()])
    all_rare_RTs  = np.hstack([x for x in rare_RTs.values()])
    med_common_RTs = [np.median(common_RTs[sID]) for sID in subject_IDs]
    med_rare_RTs   = [np.median(  rare_RTs[sID]) for sID in subject_IDs]
    # Plot cumilative histograms.
    bins = np.geomspace(100,2000,50)
    c_hist, x = np.histogram(all_common_RT, bins)
    r_hist, x = np.histogram(all_rare_RTs  , bins)
    if fig_no: 
        plt.figure(fig_no, figsize=[3,3], clear=True)
        plt.plot(bins[1:], np.cumsum(c_hist)/len(all_common_RT), col,     label='Common')
        plt.plot(bins[1:], np.cumsum(r_hist)/len(all_rare_RTs) , col+':', label='Rare')
        plt.xscale('log') 
        plt.xlabel('Reaction time (ms)')
        plt.ylabel('Fraction of trials')
        plt.xlim(bins[0], bins[-1])
        plt.ylim(0,1)
        plt.legend()
        plt.tight_layout()
    # Test for difference in common vs rare RTs
    # print('Common sec. step RT: {:.1f} + {:.1f}ms'.format(
    #     np.mean(med_common_RTs), sem(med_common_RTs)))
    # print('Rare   sec. step RT: {:.1f} + {:.1f}ms'.format(
    #     np.mean(med_rare_RTs), sem(med_rare_RTs)))
    # stats = ttest(med_rare_RTs, med_common_RTs, paired=True)
    # print(stats[['p-val','T','dof','cohen-d']])
    if return_med:
        return med_common_RTs, med_rare_RTs

def _latencies(event_times_A, event_times_B):
    'Evaluate the latency between each event A and the first event B that occurs afterwards.'                
    latencies = np.outer(event_times_B, np.ones(len(event_times_A))) - \
                np.outer(np.ones(len(event_times_B)), event_times_A)
    latencies[latencies <= 0] = np.inf
    latencies = np.min(latencies,0)
    return latencies

#----------------------------------------------------------------------------------
# Out of sequence press analyses.
#----------------------------------------------------------------------------------

def invalid_press_analysis(sessions, fig_no=1, gauss_SD=10, file_name=None,
                           return_rates=False):
    '''Analysis of presses that are invalid because they occur at the wrong 
    stage of the trial.  Plots the number of invalid presses as a function
    of trial number, the distribution of invalid presses per trial across 
    subjects and the latencies of invalid presses at the second step. 
    Invalid presses are seperated by whether the occured during the first
    or second step response periods, and at the second step by whether they
    are to a first step key or the wrong second step key. If a file name is 
    provided the number of invalid presses of each type per trial for each
    subjects is saved as a CSV.
    '''
    all_n_ip = {'f_s':[], # n presses to second step key during first step period.
                's_f':[], # n presses to first step key (up/down) during second step period.
                's_s':[], # n presses to incorrect second step key during second step period. 
                's_c':[], # n presses to incorrect second step key following common transition
                's_r':[]} # n presses to incorrect second step key following rare transition.   
    all_ip_lat = {'s_f':[], 's_s':[], 's_c':[], 'c_c':[], 'r_c':[]} # Press latencies.
    total_ip = np.zeros(len(sessions))
    for i,session in enumerate(sessions):
        n_ip, ip_latencies = _session_ip(session)
        for k,v in n_ip.items():
            all_n_ip[k].append(v)
            total_ip[i] += np.sum(v)
        for k,v in ip_latencies.items():
            all_ip_lat[k] += v
    # Calculate cross subject mean and SEM of smoothed incorrect press counts by trial.
    all_n_ip  = {k: np.vstack(v) for k, v in all_n_ip.items()}
    filt_n_ip = {k: gaussian_filter1d(v,gauss_SD,1) for k, v in all_n_ip.items()}
    mean_ip = {k: np.mean(v, 0) for k,v in filt_n_ip.items()}
    sem_ip  = {k: sem(v, 0) for k,v in filt_n_ip.items()}
    # Subject median invalid presses.
    subject_ip_rates = {k: np.mean(v,1) for k, v in all_n_ip.items()}
    subject_ip_rates['s_t'] = subject_ip_rates['s_f'] + subject_ip_rates['s_s']
    if return_rates:
        return subject_ip_rates
    print('Median subject invalid press rates:\n'
          f'First step      : {np.median(subject_ip_rates["f_s"]) :.3f}\n'
          f'Second step tot.: {np.median(subject_ip_rates["s_t"]) :.3f}\n'
          f'Second step u/d : {np.median(subject_ip_rates["s_f"]) :.3f}\n'
          f'Second step l/r : {np.median(subject_ip_rates["s_s"]) :.3f}\n'
          f'Second step com : {np.median(subject_ip_rates["s_c"]) :.3f}\n'
          f'Second step rar : {np.median(subject_ip_rates["s_r"]) :.3f}\n')
    mr, mc = (np.median(subject_ip_rates['s_r']), np.median(subject_ip_rates['s_c']))
    M, p = sign_test(subject_ip_rates['s_r'], subject_ip_rates['s_c'])
    print(f"Median invalid press rate following Common: {mc:.3f} Rare: {mr:.3f}, sign test P value:{p:.2e}, M:{M:.2f}")
    # plotting
    x = np.arange(1,len(mean_ip['f_s'])+1)
    if fig_no: plt.figure(fig_no, figsize=[3,7]).clf()
    keys, colors = ['f_s', 's_f', 's_s'], ['b', 'r', 'k']
    plt.subplot(3,1,1)
    for k, col in zip(keys, colors):
        plt.fill_between(x, mean_ip[k]-sem_ip[k], mean_ip[k]+sem_ip[k],
                         alpha=0.2, color=col, linewidth=0)
        plt.plot(x, mean_ip[k], col)
    plt.xlim(1, x[-1])
    plt.ylim(0,0.5)
    plt.xlabel('Trial number')
    plt.ylabel('Invalid presses/trial')
    plt.subplot(3,1,2)
    bins = np.arange(0,1.5,0.025)
    for k, col in zip(keys, colors):
        plt.hist(np.mean(all_n_ip[k],1), bins, color=col, alpha=0.5)
    plt.xlabel('Invalid presses/trial')
    plt.ylabel('# subjects')
    plt.subplot(3,1,3)
    plt.loglog(subject_ip_rates['s_c'], subject_ip_rates['s_r'],'.')
    plt.plot([0.001,1],[0.001,1],':k')
    plt.xlabel('Invalid following common')
    plt.ylabel('Invalid following rare')
    # Plot latencies.
    # bins = np.geomspace(20,20000,50)
    # for k, col in zip(['s_f', 's_s'], ['r', 'k']):
    #     plt.hist(all_ip_lat[k], bins, alpha=0.5, color=col)
    # median_correct_RT = np.median(all_ip_lat['s_c'])
    # plt.axvline(median_correct_RT, color='k', linestyle=':')
    # plt.xscale('log')
    # plt.xlabel('latency (ms)')
    # plt.ylabel('# presses')
    plt.tight_layout()
    if file_name: # Save data as CSV
        df = pd.DataFrame(subject_ip_rates)
        df.insert(0,'subject_IDs', [s.subject_ID for s in sessions])
        df.to_csv(file_name+'.csv', index=False)

def _session_ip(session, plot_latencies=False):
    '''Return the number of out of sequence presses at the first and
    second step for one session.'''
    n_ip = {'f_s' : np.zeros(session.n_trials), # n presses to second step key during first step period.
             's_f' : np.zeros(session.n_trials), # n presses to first step key (up/down) during second step period.
             's_s' : np.zeros(session.n_trials), # n presses to incorrect second step key during second step period.      
             's_c' : np.zeros(session.n_trials), # n presses to incorrect second step key following common transition
             's_r' : np.zeros(session.n_trials)} # n presses to incorrect second step key following rare transition.
    ip_latencies = {'s_f':[], 's_s':[], 's_c':[], 'c_c':[], 'r_c':[]} # Press latencies.
    state = False
    common_trans = session.trial_data['transitions'] == session.blocks['trial_trans_state']
    t = -1 # Trial number
    for ev in session.events:
        if ev.name == 'trial_start':
            t += 1
            state = 'first_step'
        elif ev.name == 'second_step':
            ss_onset = ev.time
            state = 'second_step'
            incorrect_ss = 'press-right' if session.trial_data['second_steps'][t] else 'press-left'
        elif state == 'first_step':
            if ev.name in ['press-left', 'press-right']:
                n_ip['f_s'][t] += 1
            elif ev.name in ['choice-up', 'choice-down']:
                state = False
        elif state == 'second_step':
            if ev.name in ['press-up', 'press-down']:
                n_ip['s_f'][t] += 1
                ip_latencies['s_f'].append(ev.time-ss_onset)
            elif ev.name == incorrect_ss:
                n_ip['s_s'][t] += 1
                ip_latencies['s_s'].append(ev.time-ss_onset)
                if common_trans[t]:
                    n_ip['s_c'][t] += 1
                else:
                    n_ip['s_r'][t] += 1
            elif ev.name in ['choice-left', 'choice-right']:
                ip_latencies['s_c'].append(ev.time-ss_onset)
                if common_trans[t]:
                    ip_latencies['c_c'].append(ev.time-ss_onset)
                else:
                    ip_latencies['r_c'].append(ev.time-ss_onset)
                state = False
    if plot_latencies:
        plt.figure(1).clf()
        bins = np.geomspace(20,20000,50)
        plt.hist(ip_latencies['s_f'], bins, alpha=0.4, label='f errors')
        plt.hist(ip_latencies['s_s'], bins, alpha=0.4, label='s errors')
        plt.hist(ip_latencies['s_c'], bins, alpha=0.4, label='correct')
        plt.xscale('log') 
        plt.legend()
    else:
        return n_ip, ip_latencies

#----------------------------------------------------------------------------------
# Stay probability Analysis
#----------------------------------------------------------------------------------

def stay_probability_analysis(sessions, ebars='SEM', selection='xtr', fig_no=1, 
                              by_trans=False, ylim=[0.,1.02], trial_mask=None, title=None, block_type='all'):
    '''Stay probability analysis.'''
    assert ebars in [None, 'SEM', 'SD'], 'Invalid error bar specifier.'
    n_sessions = len(sessions)
    all_n_trials, all_n_stay = (np.zeros([n_sessions,12]), np.zeros([n_sessions,12]))
    for i, session in enumerate(sessions):
        trial_select = session.select_trials(selection, block_type=block_type)
        if trial_mask:
            trial_select = trial_select & trial_mask[i]
        trial_select_A = trial_select &  session.blocks['trial_trans_state']
        trial_select_B = trial_select & ~session.blocks['trial_trans_state']
        #Eval total trials and number of stay trial for A and B blocks.
        all_n_trials[i,:4] , all_n_stay[i,:4]  = _stay_prob_analysis(session, trial_select_A)
        all_n_trials[i,4:8], all_n_stay[i,4:8] = _stay_prob_analysis(session, trial_select_B)
        # Evaluate combined data.
        all_n_trials[i,8:] = all_n_trials[i,:4] + all_n_trials[i,[5,4,7,6]]
        all_n_stay[i,8:] = all_n_stay[i,:4] + all_n_stay[i,[5,4,7,6]]
    if not ebars: # Don't calculate cross-animal error bars.
        mean_stay_probs = np.nanmean(all_n_stay / all_n_trials, 0)
        y_err  = np.zeros(12)
    else:
        session_sIDs = np.array([s.subject_ID for s in sessions])
        unique_sIDs = list(set(session_sIDs))
        n_subjects = len(unique_sIDs)
        per_subject_stay_probs = np.zeros([n_subjects,12])
        for i, sID in enumerate(unique_sIDs):
            session_mask = session_sIDs == sID # True for sessions with correct animal ID.
            per_subject_stay_probs[i,:] = sum(all_n_stay[session_mask,:],0) / sum(all_n_trials[session_mask,:],0)
        mean_stay_probs = np.nanmean(per_subject_stay_probs, 0)
        if ebars == 'SEM':
            y_err = ut.nansem(per_subject_stay_probs, 0)
        else:
            y_err = np.nanstd(per_subject_stay_probs, 0)
    if fig_no:
        if by_trans: # Plot seperately by transition block type.
            plt.figure(fig_no).clf()
            plt.subplot(1,3,1)
            plt.bar(np.arange(1,5), mean_stay_probs[:4], yerr=y_err[:4])
            plt.ylim(ylim)
            plt.xlim(0.75,5)
            plt.title('A transitions normal.', fontsize='small')
            plt.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
            plt.ylabel('Stay Probability')
            plt.subplot(1,3,2)
            plt.bar(np.arange(1,5), mean_stay_probs[4:8], yerr=y_err[4:8])
            plt.ylim(ylim)
            plt.xlim(0.75,5)
            plt.title('B transitions normal.', fontsize='small')
            plt.xticks([1.5,2.5,3.5,4.5],['1/A', '1/B', '0/A', '0/B'])
            plt.subplot(1,3,3)
            plt.title('Combined.', fontsize='small')
            if title: plt.suptitle(title)
        else:
            plt.figure(fig_no, figsize=[2.1,2.2]).clf()
            if title: plt.title(title)
        plt.bar(np.arange(4), mean_stay_probs[8:], zorder=0)
        sns.swarmplot(data=per_subject_stay_probs[:,8:], color='k', size=2, zorder=0.5)
        #sns.barplot(data=per_subject_stay_probs[:,8:], color='b', zorder=0)
        plt.errorbar(np.arange(4), mean_stay_probs[8:], yerr=y_err[8:], ecolor='r', ls='none')
        plt.ylim(ylim)
        plt.xlim(-0.6,3.6)
        plt.xticks(np.arange(4),['1/N', '1/R', '0/N', '0/R'])
        plt.ylabel('Stay probability')
        plt.tight_layout()
    else:
        return per_subject_stay_probs[:,8:]

def _stay_prob_analysis(session, trial_select):
    'Analysis for stay probability plots using binary mask to select trials.'
    choices, transitions, outcomes = session.unpack_trial_data('CTO', bool)
    stay = choices[1:] == choices[:-1]
    transitions, outcomes, trial_select = (transitions[:-1], outcomes[:-1], trial_select[:-1])
    stay_go_by_type = [stay[( outcomes &  transitions) & trial_select],  # A transition, rewarded.
                       stay[( outcomes & ~transitions) & trial_select],  # B transition, rewarded.
                       stay[(~outcomes &  transitions) & trial_select],  # A transition, not rewarded.
                       stay[(~outcomes & ~transitions) & trial_select]]  # B transition, not rewarded.
    n_trials_by_type = [len(s) for s in stay_go_by_type]
    n_stay_by_type =   [sum(s) for s in stay_go_by_type]
    return n_trials_by_type, n_stay_by_type

#----------------------------------------------------------------------------------
# Analysis of timecourse of emergence of repeating rewarded trials.
#----------------------------------------------------------------------------------

def outcome_effect_timecourse(sessions, smooth_SD=5, fig_no=1):
    '''Analysis of how quickly the reinforcing effect of outcome emerges 
    across the first session.  Plots the probability of repeating choice
    following rewarded and non rewarded trials as a function of trial number
    using specified gaussian smoothing.
    Also plots the P values as a function of number of trials included in analysis
    for the difference in proportion of stays following reward and non-reward, to see
    how quickly a significant difference emerges.
    '''
    # Calculate probability of staying following reward and non reward across session.
    stay_nonrew = np.full([len(sessions), sessions[0].n_trials-1], np.nan)
    stay_reward = np.full([len(sessions), sessions[0].n_trials-1], np.nan)
    init_stay_reward = np.zeros([len(sessions),20])
    init_stay_nonrew = np.zeros([len(sessions),20])
    for i, session in enumerate(sessions):
        choices, outcomes = session.unpack_trial_data('CO',bool)
        stay = choices[1:] == choices[:-1]
        stay_reward[i, outcomes[:-1]] = stay[outcomes[:-1]]
        stay_nonrew[i,~outcomes[:-1]] = stay[~outcomes[:-1]]
        init_stay_reward[i,:] = stay_reward[i,:][~np.isnan(stay_reward[i,:])][:20]
        init_stay_nonrew[i,:] = stay_nonrew[i,:][~np.isnan(stay_nonrew[i,:])][:20]
    stay_reward_filt = ut.nanGaussfilt1D(stay_reward, smooth_SD, 1)
    stay_nonrew_filt = ut.nanGaussfilt1D(stay_nonrew, smooth_SD, 1)
    mean_reward = np.mean(stay_reward_filt,0)
    mean_nonrew = np.mean(stay_nonrew_filt,0)
    sem_reward = sem(stay_reward_filt,0)
    sem_nonrew = sem(stay_nonrew_filt,0)
    # Calculate P values for difference in stay probs as function of trial number.
    sum_cum_stay_reward = np.sum(np.nancumsum(stay_reward,1),0)
    sum_cum_stay_nonrew = np.sum(np.nancumsum(stay_nonrew,1),0)
    sum_cum_n_reward = np.sum(np.cumsum(~np.isnan(stay_reward),1),0)
    sum_cum_n_nonrew = np.sum(np.cumsum(~np.isnan(stay_nonrew),1),0)
    p_vals = np.zeros(10)
    for i in range(10):
        p_vals[i] = proportions_ztest(
            [sum_cum_stay_reward[i], sum_cum_stay_nonrew[i]], 
            [sum_cum_n_reward[i]   , sum_cum_n_nonrew[i]])[1]
    # Plotting.
    x = np.arange(1,len(mean_reward)+1)
    plt.figure(fig_no, figsize=[6,3]).clf()
    plt.subplot(1,2,1)
    plt.fill_between(x, mean_reward-sem_reward, mean_reward+sem_reward, alpha=0.2, color='b', linewidth=0)
    plt.fill_between(x, mean_nonrew-sem_nonrew, mean_nonrew+sem_nonrew, alpha=0.2, color='r', linewidth=0)
    plt.plot(x, mean_reward, 'b')
    plt.plot(x, mean_nonrew, 'r')
    plt.xlim(1,300)
    plt.xlabel('Trial number')
    plt.ylabel('Stay probability')
    plt.subplot(1,2,2)
    plt.semilogy(np.arange(1,11),p_vals,'.-k')
    plt.plot([0,11],[0.05,0.05],':k')
    plt.xlabel('Trial number')
    plt.ylabel('P value')
    plt.xlim(0.5,10.5)
    plt.tight_layout()    

#----------------------------------------------------------------------------------
# Functions called by session and experiment classes.
#----------------------------------------------------------------------------------

def plot_subject(exp, sID, day_range=[0, np.inf]):
    subject_sessions =  exp.get_sessions(sID, 'all')
    if hasattr(subject_sessions[0], 'day'):
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.day)
        sorted_sessions = [s for s in sorted_sessions if
                           s.day >= day_range[0] and s.day <= day_range[1]]
    else:
        sorted_sessions = sorted(subject_sessions, key=lambda x: x.number)
    n_sessions = len(sorted_sessions)
    plt.figure(sID)
    for i,session in enumerate(sorted_sessions):
        plt.subplot(n_sessions, 1, i+1)
        session_plot(session, ylabel=False)
        plt.ylabel(session.subject_ID)
        if hasattr(session, 'day'):
            plt.ylabel(session.day)
        else:
            plt.ylabel(session.number)
    plt.suptitle('Subject ID'.format(session.subject_ID))

def plot_session(session, fig_no=1):
    'Plot data from a single session.'
    plt.figure(fig_no, figsize=[7.5, 1.8]).clf()
    session_plot(session)